import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:pytorch_mobile/pytorch_mobile.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(DigitApp(camera: cameras.first));
}

class DigitApp extends StatelessWidget {
  final CameraDescription camera;
  const DigitApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ShuffledFusionNet',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(),
      home: DigitScreen(camera: camera),
    );
  }
}

class DigitScreen extends StatefulWidget {
  final CameraDescription camera;
  const DigitScreen({super.key, required this.camera});

  @override
  State<DigitScreen> createState() => _DigitScreenState();
}

class _DigitScreenState extends State<DigitScreen> {
  late CameraController _controller;
  late Model _model;
  bool _modelLoaded = false;
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
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _model = await PyTorchMobile.loadModel('assets/shuffled_fusion_net.pt');
      setState(() => _modelLoaded = true);
    } catch (e) {
      debugPrint('Model load error: $e');
    }
  }

  /// 预处理：拍摄照片 -> MNIST 28x28 格式
  Float64List _preprocess(CameraImage image) {
    // Convert YUV420 to RGB image
    final imgLib = img.Image(width: image.width, height: image.height);

    // Extract Y plane (grayscale)
    final yPlane = image.planes[0];
    final bytes = yPlane.bytes;

    // Create grayscale image
    for (int y = 0; y < image.height; y++) {
      for (int x = 0; x < image.width; x++) {
        final idx = y * image.width + x;
        final pixel = bytes[idx];
        imgLib.setPixelRgba(x, y, pixel, pixel, pixel, 255);
      }
    }

    // Crop center 250x250
    final cropSize = 250;
    final cx = (image.width - cropSize) ~/ 2;
    final cy = (image.height - cropSize) ~/ 2;
    final cropped = img.copyCrop(imgLib, x: cx, y: cy, width: cropSize, height: cropSize);

    // Invert colors (white bg -> black bg)
    for (int y = 0; y < cropSize; y++) {
      for (int x = 0; x < cropSize; x++) {
        final p = cropped.getPixel(x, y);
        final inv = 255 - p.r;
        cropped.setPixelRgba(x, y, inv, inv, inv, 255);
      }
    }

    // Resize to 28x28
    final resized = img.copyResize(cropped, width: 28, height: 28);

    // Convert to float array with MNIST normalization
    final data = Float64List(28 * 28);
    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        final pixel = resized.getPixel(x, y).r / 255.0;
        data[y * 28 + x] = (pixel - 0.1307) / 0.3081;
      }
    }
    return data;
  }

  Future<void> _captureAndPredict() async {
    if (_processing || !_modelLoaded) return;
    setState(() => _processing = true);

    try {
      final image = await _controller.takePicture();
      final bytes = await File(image.path).readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded != null) {
        // Preprocess
        final input = _preprocessFromImage(decoded);

        // Run inference
        final output = await _model.forward(input);
        final scores = output as List<double>;

        // Softmax
        final maxScore = scores.reduce((a, b) => a > b ? a : b);
        final exps = scores.map((s) => _exp(s - maxScore)).toList();
        final sum = exps.reduce((a, b) => a + b);
        final probs = exps.map((e) => e / sum).toList();

        final pred = probs.indexOf(probs.reduce((a, b) => a > b ? a : b));
        setState(() {
          _prediction = pred;
          _confidence = probs[pred];
        });
      }
    } catch (e) {
      debugPrint('Prediction error: $e');
    }

    setState(() => _processing = false);
  }

  Float64List _preprocessFromImage(img.Image image) {
    // Grayscale
    final gray = img.grayscale(image);
    // Resize to 28x28
    final resized = img.copyResize(gray, width: 28, height: 28);

    final data = Float64List(28 * 28);
    for (int y = 0; y < 28; y++) {
      for (int x = 0; x < 28; x++) {
        final pixel = 1.0 - (resized.getPixel(x, y).r / 255.0); // invert
        data[y * 28 + x] = (pixel - 0.1307) / 0.3081;
      }
    }
    return data;
  }

  double _exp(double x) => x > 50 ? 1e20 : (x < -50 ? 0 : _expFast(x));

  // Simple exp approximation
  double _expFast(double x) {
    double result = 1.0;
    double term = 1.0;
    for (int i = 1; i <= 20; i++) {
      term *= x / i;
      result += term;
    }
    return result;
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
          // Camera preview
          CameraPreview(_controller),

          // Guide overlay
          Center(
            child: Container(
              width: 250,
              height: 250,
              decoration: BoxDecoration(
                border: Border.all(color: Colors.green, width: 2),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(
                child: Text('Place digit here',
                    style: TextStyle(color: Colors.green, fontSize: 12)),
              ),
            ),
          ),

          // Top bar
          Positioned(
            top: 50, left: 0, right: 0,
            child: Container(
              padding: const EdgeInsets.all(16),
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
                decoration: const BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white,
                ),
                child: _processing
                    ? const CircularProgressIndicator()
                    : const Icon(Icons.camera, size: 35, color: Colors.black),
              ),
            ),
          ),

          if (!_modelLoaded)
            const Center(child: CircularProgressIndicator(color: Colors.white)),
        ],
      ),
    );
  }
}
