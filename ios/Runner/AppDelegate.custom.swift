import UIKit
import Flutter
import LibTorch_Lite

@main
@objc class AppDelegate: FlutterAppDelegate {

    private var module: TorchModule?

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        GeneratedPluginRegistrant.register(with: self)

        // Load model
        loadModel()

        // Register inference channel
        if let controller = window?.rootViewController as? FlutterViewController {
            let channel = FlutterMethodChannel(
                name: "com.digit.app/inference",
                binaryMessenger: controller.binaryMessenger
            )
            channel.setMethodCallHandler { [weak self] call, result in
                if call.method == "predict" {
                    guard let args = call.arguments as? [String: Any],
                          let inputList = args["input"] as? [Double] else {
                        result(FlutterError(code: "INVALID_ARGS", message: nil, details: nil))
                        return
                    }
                    let input = inputList.map { Float($0) }
                    if let pred = self?.predict(input: input) {
                        result(["digit": pred.digit, "confidence": pred.confidence])
                    } else {
                        result(FlutterError(code: "FAILED", message: "Model not loaded", details: nil))
                    }
                } else {
                    result(FlutterMethodNotImplemented)
                }
            }
        }
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    private func loadModel() {
        guard let path = Bundle.main.path(forResource: "shuffled_fusion_net",
                                          ofType: "pt") else {
            print("[DigitApp] Model file not found in bundle")
            return
        }
        module = TorchModule(fileAtPath: path)
        if module != nil {
            print("[DigitApp] Model loaded: 39K params, 99.07%")
        }
    }

    private func predict(input: [Float]) -> (digit: Int, confidence: Float)? {
        guard let module = module else { return nil }

        // Create 4D tensor [1, 1, 28, 28]
        guard let tensor = TorchTensor.new(
            withShape: [1, 1, 28, 28],
            data: input
        ) else { return nil }

        // Forward pass -> [1, 10] logits
        guard let output = module.forward(withInput: tensor) else { return nil }

        var logits = [Float](repeating: 0, count: 10)
        output.dataCopy(to: &logits, length: 10 * MemoryLayout<Float>.size)

        // Softmax
        let maxVal = logits.max() ?? 0
        let expVals = logits.map { exp($0 - maxVal) }
        let sum = expVals.reduce(0, +)
        let probs = expVals.map { $0 / sum }

        guard let maxIdx = probs.enumerated().max(by: { $0.element < $1.element }) else {
            return nil
        }
        return (digit: maxIdx.offset, confidence: maxIdx.element)
    }
}
