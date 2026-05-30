import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'cnn/shuffled_fusion_net.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final model = ShuffledFusionNet();
  await model.loadWeights();
  runApp(DigitApp(camera: cameras.first, model: model));
}

class DigitApp extends StatelessWidget {
  final CameraDescription camera;
  final ShuffledFusionNet model;
  const DigitApp({super.key, required this.camera, required this.model});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ShuffledFusionNet',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue, brightness: Brightness.dark),
      ),
      home: DigitScreen(camera: camera, model: model),
    );
  }
}

class DigitScreen extends StatefulWidget {
  final CameraDescription camera;
  final ShuffledFusionNet model;
  const DigitScreen({super.key, required this.camera, required this.model});

  @override
  State<DigitScreen> createState() => _DigitScreenState();
}

class _DigitScreenState extends State<DigitScreen> {
  late CameraController _controller;
  int? _prediction;
  double? _confidence;
  bool _processing = false;

  @override
  void initState() {
    super.initState();
    _controller = CameraController(widget.camera, ResolutionPreset.high);
    _controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
    });
  }

  /// 预处理：拍摄 -> 灰度 -> 反转 -> 缩放 28x28 -> 归一化
  Float32List _preprocess(img.Image image) {
    final gray = img.grayscale(image);
    final resized = img.copyResize(gray, width: 28, height: 28);

    final data = Float32List(28 * 28);
    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        final p = resized.getPixel(x, y);
        final inverted = 1.0 - (p.r / 255.0);
        data[y * 28 + x] = (inverted - 0.1307) / 0.3081;
      }
    }
    return data;
  }

  Future<void> _captureAndPredict() async {
    if (_processing) return;
    setState(() => _processing = true);

    try {
      final imageFile = await _controller.takePicture();
      final bytes = await File(imageFile.path).readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded != null) {
        final input = _preprocess(decoded);

        final sw = Stopwatch()..start();
        final result = widget.model.predict(input);
        sw.stop();

        setState(() {
          _prediction = result.digit;
          _confidence = result.confidence;
        });

        debugPrint('Predicted: ${result.digit} (${(result.confidence*100).toStringAsFixed(1)}%) in ${sw.elapsedMilliseconds}ms');
      }
    } catch (e) {
      debugPrint('Error: $e');
    }

    setState(() => _processing = false);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          CameraPreview(_controller),
          // Guide box
          Center(
            child: Container(
              width: 250, height: 250,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.green, width: 2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(
                child: Text('Place digit here\n  Press button',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.green, fontSize: 12)),
              ),
            ),
          ),
          // Top bar
          Positioned(
            top: 0, left: 0, right: 0,
            child: Container(
              padding: const EdgeInsets.only(top: 50, bottom: 16, left: 16, right: 16),
              color: Colors.black54,
              child: const Row(
                children: [
                  Text('ShuffledFusionNet',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                  Spacer(),
                  Text('39K | 99.07%',
                      style: TextStyle(fontSize: 12, color: Colors.white70)),
                ],
              ),
            ),
          ),
          // Result
          if (_prediction != null)
            Positioned(
              bottom: 120, left: 0, right: 0,
              child: Container(
                padding: const EdgeInsets.all(24),
                color: Colors.black87,
                child: Column(
                  children: [
                    Text('$_prediction',
                        style: const TextStyle(fontSize: 80, fontWeight: FontWeight.bold, color: Colors.white)),
                    Text('${(_confidence! * 100).toStringAsFixed(1)}%',
                        style: TextStyle(fontSize: 24,
                            color: _confidence! > 0.7 ? Colors.green : Colors.orange)),
                  ],
                ),
              ),
            ),
          // Capture button
          Positioned(
            bottom: 30, left: 0, right: 0,
            child: GestureDetector(
              onTap: _captureAndPredict,
              child: Container(
                width: 70, height: 70,
                decoration: const BoxDecoration(shape: BoxShape.circle, color: Colors.white),
                child: _processing
                    ? const Padding(padding: EdgeInsets.all(15), child: CircularProgressIndicator(strokeWidth: 3))
                    : const Icon(Icons.camera_alt, size: 35, color: Colors.black),
              ),
            ),
          ),
        ],
      ),
    );
  }
}