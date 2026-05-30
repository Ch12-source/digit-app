import UIKit
import Flutter

@main
@objc class AppDelegate: FlutterAppDelegate {
    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        GeneratedPluginRegistrant.register(with: self)

        // Register method channel for model inference
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
                    if let prediction = self?.runInference(input: input) {
                        result(["digit": prediction.digit, "confidence": prediction.confidence])
                    } else {
                        result(FlutterError(code: "INFERENCE_FAILED", message: nil, details: nil))
                    }
                } else {
                    result(FlutterMethodNotImplemented)
                }
            }
        }

        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    private func runInference(input: [Float]) -> (digit: Int, confidence: Float)? {
        // ====================================================
        // 方案1: 使用 LibTorch (需 pod 'LibTorch-Lite')
        // ====================================================
        // guard let path = Bundle.main.path(forResource: "shuffled_fusion_net",
        //                                    ofType: "pt", inDirectory: "flutter_assets/assets"),
        //       let module = TorchModule(fileAtPath: path) else { return nil }
        // let tensor = Tensor(shape: [1,1,28,28], data: input)
        // let output = module.forward(withInput: tensor)
        // var logits = [Float](repeating: 0, count: 10)
        // output?.dataCopy(to: &logits, length: 40)
        //
        // // Softmax
        // let maxVal = logits.max() ?? 0
        // let expVals = logits.map { exp($0 - maxVal) }
        // let sum = expVals.reduce(0, +)
        // let probs = expVals.map { $0 / sum }
        // let pred = probs.enumerated().max(by: { $0.element < $1.element })
        // return pred.map { ($0.offset, $0.element) }

        // ====================================================
        // 临时方案: 返回占位结果 (替换为上述LibTorch代码)
        // ====================================================
        return (digit: 0, confidence: 1.0)
    }
}
