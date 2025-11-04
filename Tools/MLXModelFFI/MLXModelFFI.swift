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

final class ModelContext {
    let modelDirectory: URL
    let model: any LanguageModel
    var cache: [KVCache]

    init(modelDirectory: URL, model: any LanguageModel) {
        self.modelDirectory = modelDirectory
        self.model = model
        self.cache = model.newCache(parameters: nil)
    }
}

// MARK: - Error handling

private var errorMessageBuffer: UnsafeMutablePointer<CChar>? = nil
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

        // 2) Instantiate model from registry by model type
        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL, modelType: baseConfig.modelType)

        // 3) Load and apply weights (handles optional per-layer quantization)
        try loadWeights(
            modelDirectory: modelDirectory,
            model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        // 4) Return opaque handle retaining the context
        let context = ModelContext(modelDirectory: modelDirectory, model: model)
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
    
    // Read from config.json
    let configurationURL = context.modelDirectory.appending(component: "config.json")
    
    guard let configData = try? Data(contentsOf: configurationURL) else {
        setFFIError("failed to read config.json")
        return -1
    }
    
    guard let configDict = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        setFFIError("config.json is not a valid JSON object")
        return -1
    }
    
    // Helper function to extract Int32 from JSON value
    func extractInt32(_ value: Any?) -> Int32? {
        guard let value else { return nil }
        if let intValue = value as? Int {
            return Int32(intValue)
        } else if let int64Value = value as? Int64 {
            return Int32(int64Value)
        } else if let doubleValue = value as? Double {
            return Int32(doubleValue)
        }
        return nil
    }
    
    // Try to extract vocab_size
    if let vocabSizeValue = extractInt32(configDict["vocab_size"]) {
        vocabSize.pointee = vocabSizeValue
    }
    
    // Try to extract hidden_size
    if let hiddenSizeValue = extractInt32(configDict["hidden_size"]) {
        hiddenSize.pointee = hiddenSizeValue
    }
    
    // Try to extract head_dim
    if let headDimValue = extractInt32(configDict["head_dim"]) {
        headDim.pointee = headDimValue
    } else {
        // Try to calculate from hidden_size and num_attention_heads
        if let hiddenSizeValue = extractInt32(configDict["hidden_size"]),
           let numHeads = extractInt32(configDict["num_attention_heads"]),
           numHeads > 0 {
            headDim.pointee = hiddenSizeValue / numHeads
        }
    }
    
    // Try to extract num_hidden_layers
    if let numLayersValue = extractInt32(configDict["num_hidden_layers"]) {
        numLayers.pointee = numLayersValue
    }
    
    // Validate that we got at least vocab_size
    guard vocabSize.pointee > 0 else {
        setFFIError("could not determine vocab_size from config")
        return -1
    }
    
    return 0 // Success
}

// MARK: - Eval API

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
        let idsArray = Array(UnsafeBufferPointer(start: ids, count: Int(idsLength)))
        let inputTokens = MLXArray(idsArray)
        let input = LMInput(tokens: inputTokens)
        let result: MLXArray = {
            do {
                let prepared = try context.model.prepare(input, cache: context.cache, windowSize: nil)
                switch prepared {
                    case .tokens(let tokens):
                        let out = context.model(tokens[text: .newAxis], cache: context.cache.isEmpty ? nil : context.cache, state: nil)
                        return out.logits
                    case .logits(let out):
                        return out.logits
                }
            } catch {
                setFFIError("eval failed: \(error)")
                return MLXArray.zeros([1, 1, 1], type: Float32.self)
            }
        }()

        // Convert logits to float32 if needed
        let float32Logits = result.dtype == .float32 ? result : result.asType(.float32)
        
        // Get the logits shape - typically [batch, seq_len, vocab_size] or [seq_len, vocab_size]
        // For single token, we want the last position's logits
        let logitsArray: MLXArray
        let shape = float32Logits.shape
        // [batch, seq_len, vocab_size] -> take last position
        let batchSize = shape[0]
        let seqLen = shape[1]
        guard batchSize > 0 && seqLen > 0 else {
            setFFIError("invalid logits shape: \(shape)")
            return -1
        }
        logitsArray = float32Logits[0, -1, 0...]
        // Get vocabulary size
        let vocabSize = logitsArray.size
        guard vocabSize > 0 else {
            setFFIError("invalid vocab size: \(vocabSize)")
            return -1
        }
        
        // Copy logits to output buffer (C++ side is responsible for allocating enough memory)
        let logitsData = logitsArray.asData(access: .copy)
        logitsData.data.withUnsafeBytes { bytes in
            let floatBytes = bytes.bindMemory(to: Float.self)
            logits.initialize(from: floatBytes.baseAddress!, count: vocabSize)
        }
        
        return 0 // Success
        
    } catch {
        setFFIError("eval failed: \(error)")
        return -1
    }
}

// MARK: - Cache I/O API

@_cdecl("mlx_cache_get_size")
public func mlx_cache_get_size(_ handle: UnsafeMutableRawPointer?) -> Int32 {
    guard let handle else {
        setFFIError("handle is null")
        return -1
    }
    
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    
    // Calculate total size of all cache arrays in fp16 (2 bytes per element)
    var totalSize = 0
    for cache in context.cache {
        for array in cache.state {
            // Convert to fp16 if needed
            let fp16Array = array.dtype == .float16 ? array : array.asType(.float16)
            fp16Array.eval()
            totalSize += fp16Array.size * 2 // 2 bytes per fp16 element
        }
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
    
    guard let buffer else {
        setFFIError("buffer is null")
        return -1
    }
    
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    
    var offset = 0
    let bufferPtr = buffer.assumingMemoryBound(to: UInt8.self)
    
    for cache in context.cache {
        for array in cache.state {
            // Convert to fp16 if needed
            let fp16Array = array.dtype == .float16 ? array : array.asType(.float16)
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
    
    guard let buffer else {
        setFFIError("buffer is null")
        return -1
    }
    
    let context = Unmanaged<ModelContext>.fromOpaque(handle).takeUnretainedValue()
    
    // Check if cache has any state - if empty, we can't determine shapes
    guard !context.cache.isEmpty else {
        setFFIError("cache is empty, cannot determine shapes for write")
        return -1
    }
    
    // Check if all caches have state
    for cache in context.cache {
        if cache.state.isEmpty {
            setFFIError("cache state is empty, cannot determine shapes for write")
            return -1
        }
    }
    
    var offset = 0
    let bufferPtr = buffer.assumingMemoryBound(to: UInt8.self)
    
    // We need to restore cache state in the same order as reading
    // First, collect all shapes and sizes
    var cacheShapes: [[[Int]]] = []
    var cacheSizes: [[Int]] = []
    
    for cache in context.cache {
        var shapes: [[Int]] = []
        var sizes: [Int] = []
        for array in cache.state {
            let fp16Array = array.dtype == .float16 ? array : array.asType(.float16)
            fp16Array.eval()
            shapes.append(fp16Array.shape)
            sizes.append(fp16Array.size * 2) // 2 bytes per fp16 element
        }
        cacheShapes.append(shapes)
        cacheSizes.append(sizes)
    }
    
    // Now restore each cache using the saved shapes
    for cacheIndex in 0..<context.cache.count {
        var cacheState: [MLXArray] = []
        let shapes = cacheShapes[cacheIndex]
        let sizes = cacheSizes[cacheIndex]
        
        for (arrayIndex, shape) in shapes.enumerated() {
            let expectedSize = sizes[arrayIndex]
            
            guard offset + expectedSize <= Int(bufferSize) else {
                setFFIError("buffer too small for cache data")
                return -1
            }
            
            // Create Data from buffer - copy the data
            let data = Data(bytes: bufferPtr.advanced(by: offset), count: expectedSize)
            
            // Reconstruct MLXArray from fp16 data
            let restoredArray = MLXArray(data, shape, type: Float16.self)
            cacheState.append(restoredArray)
            
            offset += expectedSize
        }
        
        // Restore cache state - directly modify the cache in the array
        context.cache[cacheIndex].state = cacheState
    }
    
    guard offset == Int(bufferSize) else {
        setFFIError("buffer size mismatch: expected \(bufferSize), read \(offset)")
        return -1
    }
    
    return 0
}


