//
//  Rwkv7.swift
//  mlx-swift-examples
//
//  Created by Molly Sophia on 2025/10/29.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXRandom

public struct Rwkv7Configuration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var normEps: Float
    var headDim: Int
    var numHiddenLayers: Int
    var aLowRankDim: Int
    var vLowRankDim: Int
    var gateLowRankDim: Int
    var decayLowRankDim: Int
    private let _tieWordEmbeddings: Bool?
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? false }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case headDim = "head_dim"
        case numHiddenLayers = "num_hidden_layers"
        case aLowRankDim = "a_low_rank_dim"
        case vLowRankDim = "v_low_rank_dim"
        case gateLowRankDim = "gate_low_rank_dim"
        case decayLowRankDim = "decay_low_rank_dim"
        case _tieWordEmbeddings = "tie_word_embeddings"
    }
}

private class LayerNormPerHead: Module, UnaryLayer {
    let weight: MLXArray
    let bias: MLXArray
    let eps: Float

    public init(headDim: Int, numHeads: Int, eps: Float = 64e-5) {
        self.weight = MLXArray.ones([numHeads, headDim])
        self.bias = MLXArray.zeros([numHeads, headDim])
        self.eps = eps
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return self.weight * MLXFast.layerNorm(x, eps: eps) + self.bias
    }
}

private class TokenShift: Module {
    func callAsFunction(_ x: MLXArray, _ cache: MLXArray?) -> MLXArray {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))

        let cache = cache ?? MLXArray.zeros([B, 1, D], dtype: x.dtype)

        if L > 1 {
            return concatenated([cache, x[0..., ..<(L - 1), 0...]], axis: 1)
        } else {
            return cache
        }
    }
}

private func compileAddCMul() -> @Sendable (MLXArray, MLXArray, MLXArray) -> MLXArray {
    compile(shapeless: true) { x, y, z in
        x + y * z
    }
}

private func compileL2Norm() -> @Sendable (MLXArray) -> MLXArray {
    compile(shapeless: true) { x in
        x / maximum(norm(x, axis: -1, keepDims: true), MLXArray(1e-7))
    }
}

private func makeWkv7Kernel() -> MLXFast.MLXFastKernel? {
    let source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / H;
        auto h_idx = n % H;
        constexpr int n_per_t = D / 32;

        // [B, T, H, D]
        auto r_ = r + b_idx * T * H * D + h_idx * D;
        auto w_ = w + b_idx * T * H * D + h_idx * D;
        auto k_ = k + b_idx * T * H * D + h_idx * D;
        auto v_ = v + b_idx * T * H * D + h_idx * D;
        auto a_ = a + b_idx * T * H * D + h_idx * D;
        auto b_ = b + b_idx * T * H * D + h_idx * D;
        y += b_idx * T * H * D + h_idx * D;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // state_in, state_out: [B, H, D, D]
        auto i_state = state_in  + (n * D + dv_idx) * D;
        auto o_state = state_out + (n * D + dv_idx) * D;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }}

        for (int t = 0; t < T; ++t) {{
          float sa = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            sa += state[i] * a_[s_idx];
            state[i] = state[i] * w_[s_idx];
          }}
          sa = simd_sum(sa);

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {{
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * v_[dv_idx] + sa * b_[s_idx];
            out += state[i] * r_[s_idx];
          }}
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {{
            y[dv_idx] = static_cast<InT>(out);
          }}

          // Increment data pointers to next time step
          r_ += H * D;
          w_ += H * D;
          k_ += H * D;
          v_ += H * D;
          a_ += H * D;
          b_ += H * D;
          y  += H * D;
        }}
        for (int i = 0; i < n_per_t; ++i) {{
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }}
    """

    return MLXFast.metalKernel(
        name: "wkv7_kernel",
        inputNames: ["r", "w", "k", "v", "a", "b", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: source
    )
}

private final class Wkv7KernelManager: @unchecked Sendable {
    static let shared = Wkv7KernelManager()

    let wkv7Kernel: MLXFast.MLXFastKernel?

    private init() {
        wkv7Kernel = makeWkv7Kernel()
    }
}

func wkv7Kernel(
    r: MLXArray,
    w: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    state: MLXArray,
) -> (MLXArray, MLXArray) {
    let (B, T, H, D) = (r.dim(0), r.dim(1), r.dim(2), r.dim(3))
    let inputType = r.dtype

    guard let kernel = Wkv7KernelManager.shared.wkv7Kernel else {
        fatalError("Wkv7 kernel not available")
    }

    let outputs = kernel(
        [r, w, k, v, a, b, state, T],
        template: [
            ("InT", inputType),
            ("H", H),
            ("D", D),
        ],
        grid: (32, D, B * H),
        threadGroup: (32, 4, 1),
        outputShapes: [[B, T, H, D], state.shape],
        outputDTypes: [inputType, inputType]
    )

    return (outputs[0], outputs[1])
}

private func Wkv7StepOps(r: MLXArray, w: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
    let sab = matmul(matmul(state, a[.ellipsis, .newAxis]), b[.ellipsis, .newAxis, 0...])
    let updatedState = state * w[.ellipsis, .newAxis, 0...] +
        matmul(v[.ellipsis, .newAxis], k[.ellipsis, .newAxis, 0...]) +
        sab
    let y = matmul(updatedState, r[.ellipsis, .newAxis])
    return (y, updatedState)
}

private class LoRA: Module, UnaryLayer {
    fileprivate let lora: [UnaryLayer]
    let activation: UnaryLayer

    public init(inputDim: Int, outputDim: Int, lowRankDim: Int, bias: Bool = true, activation: String) {
        if activation == "tanh" {
            self.activation = Tanh()
        } else if activation == "sigmoid" {
            self.activation = Sigmoid()
        } else if activation == "relu" {
            self.activation = ReLU()
        } else if activation == "identity" {
            self.activation = Identity()
        } else {
            fatalError("Invalid activation: \(activation)")
        }

        self.lora = [
            Linear(inputDim, lowRankDim, bias: false),
            self.activation,
            Linear(lowRankDim, outputDim, bias: bias)
        ]
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return self.lora[2](self.lora[1](self.lora[0](x)))
    }
}

private class Rwkv7TimeMixing: Module {
    let config: Rwkv7Configuration
    let layerIdx: Int
    let hiddenSize: Int
    let headDim: Int
    let numHeads: Int
    let aLowRankDim: Int
    let vLowRankDim: Int
    let gateLowRankDim: Int
    let decayLowRankDim: Int

    let tokenShift: TokenShift

    @ModuleInfo(key: "x_r") var xR: MLXArray
    @ModuleInfo(key: "x_w") var xW: MLXArray
    @ModuleInfo(key: "x_k") var xK: MLXArray
    @ModuleInfo(key: "x_v") var xV: MLXArray
    @ModuleInfo(key: "x_a") var xA: MLXArray
    @ModuleInfo(key: "x_g") var xG: MLXArray

    @ModuleInfo(key: "k_k") var kK: MLXArray
    @ModuleInfo(key: "k_a") var kA: MLXArray
    @ModuleInfo(key: "r_k") var rK: MLXArray

    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "g_norm") var gNorm: LayerNormPerHead
    @ModuleInfo(key: "w_lora") var wLora: LoRA
    @ModuleInfo(key: "a_lora") var aLora: LoRA
    @ModuleInfo(key: "g_lora") var gLora: LoRA
    @ModuleInfo(key: "v_lora") var vLora: LoRA?

    init(_ config: Rwkv7Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx

        self.hiddenSize = config.hiddenSize
        self.headDim = config.headDim
        self.numHeads = hiddenSize / headDim
        self.aLowRankDim = config.aLowRankDim
        self.vLowRankDim = config.vLowRankDim
        self.gateLowRankDim = config.gateLowRankDim
        self.decayLowRankDim = config.decayLowRankDim

        self._xR.wrappedValue = MLXArray.zeros([1, 1, hiddenSize])
        self._xW.wrappedValue = MLXArray.zeros([1, 1, hiddenSize])
        self._xK.wrappedValue = MLXArray.zeros([1, 1, hiddenSize])
        self._xV.wrappedValue = MLXArray.zeros([1, 1, hiddenSize])
        self._xA.wrappedValue = MLXArray.zeros([1, 1, hiddenSize])
        self._xG.wrappedValue = MLXArray.zeros([1, 1, hiddenSize])
        self._kK.wrappedValue = MLXArray.zeros([numHeads, headDim])
        self._kA.wrappedValue = MLXArray.zeros([numHeads, headDim])
        self._rK.wrappedValue = MLXArray.zeros([numHeads, headDim])

        self._rProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._kProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._vProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        self._oProj.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)

        self._wLora.wrappedValue = LoRA(
            inputDim: hiddenSize, outputDim: hiddenSize, lowRankDim: decayLowRankDim, activation: "tanh"
        )
        self._aLora.wrappedValue = LoRA(
            inputDim: hiddenSize, outputDim: hiddenSize, lowRankDim: aLowRankDim, activation: "identity"
        )
        self._gLora.wrappedValue = LoRA(
            inputDim: hiddenSize, outputDim: hiddenSize, lowRankDim: gateLowRankDim, bias: false, activation: "sigmoid"
        )
        if layerIdx > 0 {
            self._vLora.wrappedValue = LoRA(
                inputDim: hiddenSize, outputDim: hiddenSize, lowRankDim: vLowRankDim, activation: "identity"
            )
        } else {
            self._vLora.wrappedValue = nil
        }

        self.tokenShift = TokenShift()
        self._gNorm.wrappedValue = LayerNormPerHead(headDim: headDim, numHeads: numHeads)
    }

    func _wkv7(r: MLXArray, w: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray, state: MLXArray?, useKernel: Bool = true) -> (MLXArray, MLXArray) {
        let (B, L) = (r.dim(0), r.dim(1))

        var state = state ?? MLXArray.zeros([B, self.numHeads, self.headDim, self.headDim], dtype: r.dtype)

        if !useKernel || Wkv7KernelManager.shared.wkv7Kernel == nil {
            var ys = Array<MLXArray>();
            for t in 0 ..< L {
                let (y, updatedState) = Wkv7StepOps(
                    r: r[0..., t, 0..., 0...],
                    w: w[0..., t, 0..., 0...],
                    k: k[0..., t, 0..., 0...],
                    v: v[0..., t, 0..., 0...],
                    a: a[0..., t, 0..., 0...],
                    b: b[0..., t, 0..., 0...],
                    state: state,
                )
                ys.append(y)
                state = updatedState
            }
            return (concatenated(ys, axis: 1).asType(r.dtype), state)
        } else {
            return wkv7Kernel(r: r, w: w, k: k, v: v, a: a, b: b, state: state)
        }
    }

    func callAsFunction(
        _ x: MLXArray, vFirst: MLXArray?, cache: KVCache?
    ) -> (MLXArray, MLXArray?) {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))

        var tokenShiftCache: MLXArray? = nil
        var stateCache: MLXArray? = nil
        if let cache = cache as? RwkvCache {
            tokenShiftCache = cache[0]
            stateCache = cache[1]
        }

        let xPrev = self.tokenShift(x, tokenShiftCache)
        let xDelta = xPrev - x

        let xR = compileAddCMul()(x, xDelta, self.xR)
        let xW = compileAddCMul()(x, xDelta, self.xW)
        let xK = compileAddCMul()(x, xDelta, self.xK)
        let xV = compileAddCMul()(x, xDelta, self.xV)
        let xA = compileAddCMul()(x, xDelta, self.xA)
        let xG = compileAddCMul()(x, xDelta, self.xG)

        let key = self.kProj(xK).reshaped(B, L, numHeads, headDim)
        let value = self.vProj(xV).reshaped(B, L, numHeads, headDim)
        let receptance = self.rProj(xR).reshaped(B, L, numHeads, headDim)
        let iclr = sigmoid(self.aLora(xA)).reshaped(B, L, numHeads, headDim)
        let gate = self.gLora(xG)

        var vFirst: MLXArray? = vFirst
        var valueFinal: MLXArray? = nil

        if self.layerIdx == 0 {
            vFirst = value
            valueFinal = value
        } else {
            let vV = sigmoid(self.vLora!(xV)).reshaped(B, L, numHeads, headDim)
            valueFinal = compileAddCMul()(value, vFirst! - value, vV)
        }

        var decay = sigmoid(self.wLora(xW).reshaped(B, L, numHeads, headDim)).asType(.float32)
        decay = exp(-0.606531 * decay).asType(receptance.dtype)
        let kK: MLXArray = compileL2Norm()(key * self.kK)
        let keyFinal = key * (1 + (iclr - 1) * self.kA)
        let a = -kK
        let b = kK * iclr

        var (out, updatedState) = _wkv7(r: receptance, w: decay, k: keyFinal, v: valueFinal!, a: a, b: b, state: stateCache)
        out = self.gNorm(out.reshaped(B, L, numHeads, headDim))
        out = (out + (receptance * keyFinal * self.rK).sum(axis: -1, keepDims: true) * valueFinal!).reshaped(B, L, D)

        if let cache = cache as? RwkvCache {
            cache[0] = x[0..., (L - 1)..., 0...]
            cache[1] = updatedState
        }

        out = self.oProj(out * gate)
        return (out, vFirst)
    }
}

private class Rwkv7ChannelMixing: Module {
    let key: Linear
    let value: Linear
    let tokenShift: TokenShift
    @ModuleInfo(key: "x_k") var xK: MLXArray

    init(_ config: Rwkv7Configuration) {
        self.key = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self.value = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        self._xK.wrappedValue = MLXArray.zeros([config.hiddenSize])
        self.tokenShift = TokenShift()
    }

    func callAsFunction(_ x: MLXArray, _ cache: KVCache?) -> MLXArray {
        let (_, L, _) = (x.dim(0), x.dim(1), x.dim(2))
        var state: MLXArray? = nil
        if let cache = cache as? RwkvCache {
            state = cache[2]
            cache[2] = x[0..., (L - 1)..., 0...]
        }
        let xPrev = self.tokenShift(x, state)
        let xx = compileAddCMul()(x, xPrev - x, self.xK)
        return self.value(reluSquared(self.key(xx)))
    }
}

private class Rwkv7Layer: Module {
    let config: Rwkv7Configuration
    let attn: Rwkv7TimeMixing
    let ffn: Rwkv7ChannelMixing
    @ModuleInfo(key: "attn_norm") var attnNorm: LayerNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: LayerNorm
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm?

    let layerIdx: Int

    init(_ config: Rwkv7Configuration, layerIdx: Int) {
        self.config = config
        self.layerIdx = layerIdx

        self.attn = Rwkv7TimeMixing(config, layerIdx: layerIdx)
        self.ffn = Rwkv7ChannelMixing(config)
        _attnNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _ffnNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps)
        if layerIdx == 0 {
            _preNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps)
        } else {
            _preNorm.wrappedValue = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray, vFirst: MLXArray?, cache: KVCache?
    ) -> (MLXArray, MLXArray?) {
        let x = self.preNorm?(x).asType(.float16) ?? x
        let (r, vFirst) = self.attn(self.attnNorm(x), vFirst: vFirst, cache: cache)
        let h = x + r
        let out = h + self.ffn(self.ffnNorm(h), cache)
        return (out, vFirst)
    }
}

private class Rwkv7ModelInner: Module {
    let args: Rwkv7Configuration
    let embeddings: Embedding

    fileprivate let layers: [Rwkv7Layer]
    let norm: LayerNorm

    init(_ config: Rwkv7Configuration) {
        self.args = config
        self.embeddings = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { Rwkv7Layer(config, layerIdx: $0) }
        self.norm = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var x = self.embeddings(inputs)
        var vFirst: MLXArray? = nil

        for (i, layer) in layers.enumerated() {
            (x, vFirst) = layer(x, vFirst: vFirst, cache: cache?[i])
        }

        return norm(x)
    }
}

public class Rwkv7Model: Module, LLMModel {
    private let model: Rwkv7ModelInner
    let args: Rwkv7Configuration

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: Rwkv7Configuration) {
        self.args = config
        self.model = Rwkv7ModelInner(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var outputs = model(inputs, cache: cache)

        if let lmHead {
            outputs = lmHead(outputs)
        } else {
            outputs = model.embeddings.asLinear(outputs)
        }

        return outputs
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        return model.layers.enumerated().map { (i, _) in
            return RwkvCache()
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitizedWeights = [String: MLXArray]()

        for (key, value) in weights {
            var new_value = value
            if value.dtype == .bfloat16 && !key.contains("embeddings") {
                new_value = value.asType(.float16)
            }

            if (key.contains("k_k") || key.contains("k_a") || key.contains("g_norm")) {
                new_value = new_value.reshaped(
                    self.args.hiddenSize / self.args.headDim, self.args.headDim
                )
            }
            sanitizedWeights[key] = new_value
        }

        return sanitizedWeights
    }
}

extension Rwkv7Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        // TODO
        return []
    }
}
