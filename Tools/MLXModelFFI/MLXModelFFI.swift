//
//  MLXModelFFI.swift
//  MLXModelFFI
//
//  Created by molly on 2025/11/4.
//

import Foundation
import MLXLMCommon
import MLXLLM
import MLX

private final class ModelContext {
    let modelDirectory: URL
    let model: any LanguageModel
    let config: FFIModelConfig
    var cache: [KVCache]

    init(modelDirectory: URL, model: any LanguageModel, config: FFIModelConfig) {
        self.modelDirectory = modelDirectory
        self.model = model
        self.config = config
        self.cache = model.newCache(parameters: nil)
    }
}

private struct FFIModelConfig {
    let vocabSize: Int
    let hiddenSize: Int
    let headDim: Int
    let numLayers: Int

    var numHeads: Int {
        hiddenSize / headDim
    }
}

private func intValue(_ value: Any?) -> Int? {
    guard let value else { return nil }
    if let intValue = value as? Int {
        return intValue
    }
    if let int64Value = value as? Int64 {
        return Int(int64Value)
    }
    if let doubleValue = value as? Double {
        return Int(doubleValue)
    }
    return nil
}

private func readFFIModelConfig(from configurationURL: URL) throws -> FFIModelConfig {
    let configData = try Data(contentsOf: configurationURL)
    guard let configDict = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        throw NSError(domain: "MLXModelFFI", code: 1, userInfo: [NSLocalizedDescriptionKey: "config.json is not a valid JSON object"])
    }

    guard let vocabSize = intValue(configDict["vocab_size"]), vocabSize > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 2, userInfo: [NSLocalizedDescriptionKey: "could not determine vocab_size from config"])
    }
    guard let hiddenSize = intValue(configDict["hidden_size"]), hiddenSize > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 3, userInfo: [NSLocalizedDescriptionKey: "could not determine hidden_size from config"])
    }
    let headDim = intValue(configDict["head_dim"]) ??
        intValue(configDict["num_attention_heads"]).map { $0 > 0 ? hiddenSize / $0 : 0 } ?? 0
    guard headDim > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 4, userInfo: [NSLocalizedDescriptionKey: "could not determine head_dim from config"])
    }
    guard let numLayers = intValue(configDict["num_hidden_layers"]), numLayers > 0 else {
        throw NSError(domain: "MLXModelFFI", code: 5, userInfo: [NSLocalizedDescriptionKey: "could not determine num_hidden_layers from config"])
    }

    return FFIModelConfig(vocabSize: vocabSize, hiddenSize: hiddenSize, headDim: headDim, numLayers: numLayers)
}

// MARK: - Error handling

nonisolated(unsafe) private var errorMessageBuffer: UnsafeMutablePointer<CChar>? = nil
private let errorQueue = DispatchQueue(label: "mlx_model_ffi.error.queue")

@inline(__always)
private func setFFIError(_ message: String) {
    errorQueue.sync {
        if let old = errorMessageBuffer {
            free(old)
            errorMessageBuffer = nil
        }
        errorMessageBuffer = strdup(message)
    }
}

@_cdecl("mlx_last_error_message")
public func mlx_last_error_message() -> UnsafePointer<CChar>? {
    return UnsafePointer(errorMessageBuffer)
}

// MARK: - FFI API

@_cdecl("mlx_initialize")
public func mlx_initialize() -> Int32 {
    // Force initialization by creating a dummy array and evaluating it
    // This triggers MLX's internal initialization and loads the metallib
    let _ = MLXArray(0.0).eval()
    return 0 // Success
}

@_cdecl("mlx_model_load")
public func mlx_model_load(_ cPath: UnsafePointer<CChar>?) -> UnsafeMutableRawPointer? {
    guard let cPath else {
        setFFIError("path is null")
        return nil
    }

    let path = String(cString: cPath)

    // Resolve model directory
    let modelDirectory = URL(filePath: path)
    guard FileManager.default.fileExists(atPath: modelDirectory.path) else {
        setFFIError("file or directory not found: \(path)")
        return nil
    }

    do {
        // 1) Read base config to get model type and quantization
        let configurationURL = modelDirectory.appending(component: "config.json")
        let baseConfig = try JSONDecoder().decode(
            BaseConfiguration.self, from: Data(contentsOf: configurationURL))
        let ffiConfig = try readFFIModelConfig(from: configurationURL)

        // 2) Instantiate model from registry by model type
        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType)

        // 3) Load and apply weights (handles optional per-layer quantization)
        try loadWeights(
            modelDirectory: modelDirectory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        // 4) Return opaque handle retaining the context
        let context = ModelContext(modelDirectory: modelDirectory, model: model, config: ffiConfig)
        return Unmanaged.passRetained(context).toOpaque()

    } catch {
        setFFIError("load failed: \(error)")
        return nil
    }
}

@_cdecl("mlx_model_release")
public func mlx_model_release(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<ModelContext>.fromOpaque(handle).release()
    GPU.clearCache()
}

// MARK: - Configuration API

@_cdecl("mlx_model_get_config")
public func mlx_model_get_config(
    _ handle: UnsafeMutableRawPointer?,
    _ vocabSize: UnsafeMutablePointer<Int32>?,
    _ hiddenSize: UnsafeMutablePointer<Int32>?,
    _ headDim: UnsafeMutablePointer<Int32>?,
    _ numLayers: UnsafeMutablePointer<Int32>?
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    guard let vocabSize, let hiddenSize, let headDim, let numLayers else {
        setFFIError("output pointers are null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()

    vocabSize.pointee = Int32(context.config.vocabSize)
    hiddenSize.pointee = Int32(context.config.hiddenSize)
    headDim.pointee = Int32(context.config.headDim)
    numLayers.pointee = Int32(context.config.numLayers)

    return 0 // Success
}

// MARK: - Eval API

private func resizeArrayBatch(_ array: MLXArray, batchSize: Int) -> MLXArray {
    let currentBatchSize = array.dim(0)
    if currentBatchSize == batchSize {
        return array
    }
    if currentBatchSize > batchSize {
        return array[0 ..< batchSize, 0...]
    }

    var extraShape = array.shape
    extraShape[0] = batchSize - currentBatchSize
    let zeros = MLXArray.zeros(extraShape, dtype: array.dtype)
    return concatenated([array, zeros], axis: 0)
}

private func ensureCacheBatchSize(_ context: ModelContext, batchSize: Int) -> Bool {
    guard batchSize > 0 else {
        setFFIError("batch size must be greater than 0")
        return false
    }

    for cacheIndex in 0..<context.cache.count {
        let state = context.cache[cacheIndex].state
        if state.isEmpty {
            continue
        }
        context.cache[cacheIndex].state = state.map { resizeArrayBatch($0, batchSize: batchSize) }
    }
    return true
}

private func evaluateModel(_ context: ModelContext, input: LMInput) throws -> MLXArray {
    let prepared = try context.model.prepare(input, cache: context.cache, windowSize: nil)
    switch prepared {
        case .tokens(let tokens):
            let modelInput: LMInput.Text
            if tokens.tokens.ndim == 1 {
                modelInput = tokens[text: .newAxis]
            } else {
                modelInput = tokens
            }
            let out = context.model(modelInput, cache: context.cache.isEmpty ? nil : context.cache, state: nil)
            return out.logits
        case .logits(let out):
            return out.logits
    }
}

private func copyLastLogits(_ result: MLXArray, expectedBatchSize: Int, to logits: UnsafeMutablePointer<Float>) -> Int32 {
    let float32Logits = result.dtype == .float32 ? result : result.asType(.float32)

    let logitsArray: MLXArray
    if float32Logits.ndim == 3 {
        let batchSize = float32Logits.dim(0)
        let seqLen = float32Logits.dim(1)
        guard batchSize == expectedBatchSize, seqLen > 0 else {
            setFFIError("invalid logits shape: \(float32Logits.shape), expected batch \(expectedBatchSize)")
            return -1
        }
        if expectedBatchSize == 1 {
            logitsArray = float32Logits[0, -1, 0...]
        } else {
            logitsArray = float32Logits[0..., -1, 0...]
        }
    } else if float32Logits.ndim == 2 && expectedBatchSize == 1 {
        let seqLen = float32Logits.dim(0)
        guard seqLen > 0 else {
            setFFIError("invalid logits shape: \(float32Logits.shape)")
            return -1
        }
        logitsArray = float32Logits[-1, 0...]
    } else {
        setFFIError("unsupported logits shape: \(float32Logits.shape)")
        return -1
    }

    let logitsCount = logitsArray.size
    guard logitsCount > 0 else {
        setFFIError("invalid logits size: \(logitsCount)")
        return -1
    }

    let logitsData = logitsArray.asData(access: .copy)
    logitsData.data.withUnsafeBytes { bytes in
        let floatBytes = bytes.bindMemory(to: Float.self)
        logits.initialize(from: floatBytes.baseAddress!, count: logitsCount)
    }

    return 0
}

@_cdecl("mlx_model_eval")
public func mlx_model_eval(
    _ handle: UnsafeMutableRawPointer?,
    _ ids: UnsafePointer<Int32>?,
    _ idsLength: Int32,
    _ logits: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    guard let ids else {
        setFFIError("ids is null")
        return -1
    }

    guard let logits else {
        setFFIError("logits output pointer is null")
        return -1
    }

    guard idsLength > 0 else {
        setFFIError("ids length must be greater than 0")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    do {
        guard ensureCacheBatchSize(context, batchSize: 1) else {
            return -1
        }

        let idsArray = Array(UnsafeBufferPointer(start: ids, count: Int(idsLength)))
        let inputTokens = MLXArray(idsArray)
        let input = LMInput(tokens: inputTokens)
        let result = try evaluateModel(context, input: input)
        return copyLastLogits(result, expectedBatchSize: 1, to: logits)

    } catch {
        setFFIError("eval failed: \(error)")
        return -1
    }
}

@_cdecl("mlx_model_eval_batch_tokens")
public func mlx_model_eval_batch_tokens(
    _ handle: UnsafeMutableRawPointer?,
    _ ids: UnsafePointer<Int32>?,
    _ batchSize: Int32,
    _ logits: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    guard let ids else {
        setFFIError("ids is null")
        return -1
    }

    guard let logits else {
        setFFIError("logits output pointer is null")
        return -1
    }

    guard batchSize > 0 else {
        setFFIError("batch size must be greater than 0")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    let batchCount = Int(batchSize)
    do {
        guard ensureCacheBatchSize(context, batchSize: batchCount) else {
            return -1
        }

        let idsArray = Array(UnsafeBufferPointer(start: ids, count: batchCount))
        let inputTokens = MLXArray(idsArray).reshaped(batchCount, 1)
        let input = LMInput(tokens: inputTokens)
        let result = try evaluateModel(context, input: input)
        return copyLastLogits(result, expectedBatchSize: batchCount, to: logits)
    } catch {
        setFFIError("batch eval failed: \(error)")
        return -1
    }
}

// MARK: - Cache I/O API

private func defaultSlotShapes(_ context: ModelContext) -> [[Int]] {
    [
        [1, 1, context.config.hiddenSize],
        [1, context.config.numHeads, context.config.headDim, context.config.headDim],
        [1, 1, context.config.hiddenSize],
    ]
}

private func slotShape(for shape: [Int]) -> [Int] {
    guard !shape.isEmpty else {
        return shape
    }
    return [1] + Array(shape.dropFirst())
}

private func slotShapes(for cache: KVCache, context: ModelContext) -> [[Int]] {
    let state = cache.state
    if state.isEmpty {
        return defaultSlotShapes(context)
    }
    return state.map { slotShape(for: $0.shape) }
}

private func elementCount(_ shape: [Int]) -> Int {
    shape.reduce(1, *)
}

private func cacheSlotSize(_ context: ModelContext, slot: Int) -> Int? {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return nil
    }

    var totalSize = 0
    for cache in context.cache {
        for shape in slotShapes(for: cache, context: context) {
            totalSize += elementCount(shape) * 2
        }
    }
    return totalSize
}

private func installSlotArray(current: MLXArray?, slot: Int, slotArray: MLXArray) -> MLXArray {
    if let current {
        let target = resizeArrayBatch(current, batchSize: max(current.dim(0), slot + 1))
        target[slot ..< (slot + 1), 0...] = slotArray.asType(target.dtype)
        return target
    }

    var shape = slotArray.shape
    shape[0] = slot + 1
    let target = MLXArray.zeros(shape, dtype: slotArray.dtype)
    target[slot ..< (slot + 1), 0...] = slotArray
    return target
}

private func readCacheSlot(_ context: ModelContext, slot: Int, buffer: UnsafeMutableRawPointer?, bufferSize: Int32) -> Int32 {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return -1
    }

    guard let buffer else {
        setFFIError("buffer is null")
        return -1
    }

    var offset = 0
    let bufferPtr = buffer.assumingMemoryBound(to: UInt8.self)

    for cache in context.cache {
        let state = cache.state
        let arrays = state.isEmpty
            ? defaultSlotShapes(context).map { MLXArray.zeros($0, dtype: .float16) }
            : state

        for array in arrays {
            let slotArray: MLXArray
            if state.isEmpty {
                slotArray = array
            } else {
                guard slot < array.dim(0) else {
                    setFFIError("slot \(slot) out of range for cache batch size \(array.dim(0))")
                    return -1
                }
                slotArray = array[slot ..< (slot + 1), 0...]
            }

            let fp16Array = slotArray.dtype == .float16 ? slotArray : slotArray.asType(.float16)
            fp16Array.eval()

            let arrayData = fp16Array.asData(access: .copy)
            let arraySize = arrayData.data.count

            guard offset + arraySize <= Int(bufferSize) else {
                setFFIError("buffer too small for cache data")
                return -1
            }

            _ = arrayData.data.withUnsafeBytes { bytes in
                memcpy(bufferPtr.advanced(by: offset), bytes.baseAddress!, arraySize)
            }
            offset += arraySize
        }
    }

    return Int32(offset)
}

private func writeCacheSlot(_ context: ModelContext, slot: Int, buffer: UnsafeRawPointer?, bufferSize: Int32) -> Int32 {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return -1
    }

    guard let buffer else {
        setFFIError("buffer is null")
        return -1
    }

    let maybeExpectedSize = cacheSlotSize(context, slot: slot)
    guard let expectedSize = maybeExpectedSize, expectedSize == Int(bufferSize) else {
        setFFIError("buffer size mismatch: expected \(maybeExpectedSize ?? -1), got \(bufferSize)")
        return -1
    }

    var offset = 0
    let bufferPtr = buffer.assumingMemoryBound(to: UInt8.self)

    for cacheIndex in 0..<context.cache.count {
        let state = context.cache[cacheIndex].state
        let shapes = slotShapes(for: context.cache[cacheIndex], context: context)
        let currentArrays: [MLXArray?] = state.isEmpty ? Array(repeating: nil, count: shapes.count) : state.map { Optional($0) }
        var cacheState: [MLXArray] = []

        for (arrayIndex, shape) in shapes.enumerated() {
            let expectedArraySize = elementCount(shape) * 2
            guard offset + expectedArraySize <= Int(bufferSize) else {
                setFFIError("buffer too small for cache data")
                return -1
            }

            let data = Data(bytes: bufferPtr.advanced(by: offset), count: expectedArraySize)
            let restoredSlot = MLXArray(data, shape, type: Float16.self)
            cacheState.append(installSlotArray(current: currentArrays[arrayIndex], slot: slot, slotArray: restoredSlot))
            offset += expectedArraySize
        }

        context.cache[cacheIndex].state = cacheState
    }

    guard offset == Int(bufferSize) else {
        setFFIError("buffer size mismatch: expected \(bufferSize), read \(offset)")
        return -1
    }

    return 0
}

private func zeroCacheSlot(_ context: ModelContext, slot: Int) -> Int32 {
    guard slot >= 0 else {
        setFFIError("slot must be non-negative")
        return -1
    }

    for cacheIndex in 0..<context.cache.count {
        let state = context.cache[cacheIndex].state
        let shapes = slotShapes(for: context.cache[cacheIndex], context: context)
        let currentArrays: [MLXArray?] = state.isEmpty ? Array(repeating: nil, count: shapes.count) : state.map { Optional($0) }
        var cacheState: [MLXArray] = []

        for (arrayIndex, shape) in shapes.enumerated() {
            let dtype = currentArrays[arrayIndex]?.dtype ?? .float16
            let zeroSlot = MLXArray.zeros(shape, dtype: dtype)
            cacheState.append(installSlotArray(current: currentArrays[arrayIndex], slot: slot, slotArray: zeroSlot))
        }

        context.cache[cacheIndex].state = cacheState
    }

    return 0
}

@_cdecl("mlx_cache_get_size")
public func mlx_cache_get_size(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    guard let totalSize = cacheSlotSize(context, slot: 0) else {
        return -1
    }
    return Int32(totalSize)
}

@_cdecl("mlx_cache_read")
public func mlx_cache_read(
    _ handle: UnsafeMutableRawPointer?,
    _ buffer: UnsafeMutableRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return readCacheSlot(context, slot: 0, buffer: buffer, bufferSize: bufferSize)
}

@_cdecl("mlx_cache_write")
public func mlx_cache_write(
    _ handle: UnsafeMutableRawPointer?,
    _ buffer: UnsafeRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return writeCacheSlot(context, slot: 0, buffer: buffer, bufferSize: bufferSize)
}

@_cdecl("mlx_cache_read_slot")
public func mlx_cache_read_slot(
    _ handle: UnsafeMutableRawPointer?,
    _ slot: Int32,
    _ buffer: UnsafeMutableRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return readCacheSlot(context, slot: Int(slot), buffer: buffer, bufferSize: bufferSize)
}

@_cdecl("mlx_cache_write_slot")
public func mlx_cache_write_slot(
    _ handle: UnsafeMutableRawPointer?,
    _ slot: Int32,
    _ buffer: UnsafeRawPointer?,
    _ bufferSize: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return writeCacheSlot(context, slot: Int(slot), buffer: buffer, bufferSize: bufferSize)
}

@_cdecl("mlx_cache_zero_slot")
public func mlx_cache_zero_slot(
    _ handle: UnsafeMutableRawPointer?,
    _ slot: Int32
) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }

    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    return zeroCacheSlot(context, slot: Int(slot))
}
