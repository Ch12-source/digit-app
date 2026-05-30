import 'dart:io';
import 'dart:typed_data';
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/services.dart';

/// Pure Dart CNN inference for ShuffledFusionNet
/// Architecture: AMKF + SGDR + CSA (39K params, 99.07% MNIST)
/// No native dependencies - runs entirely in Dart

class Tensor {
  final int n, c, h, w;
  final Float32List data; // NHWC layout for simplicity

  Tensor._(this.n, this.c, this.h, this.w, this.data);

  factory Tensor.zeros(int n, int c, int h, int w) {
    return Tensor._(n, c, h, w, Float32List(n * c * h * w));
  }

  factory Tensor.fromList(int n, int c, int h, int w, List<double> values) {
    return Tensor._(n, c, h, w, Float32List.fromList(values.map((v) => v.toDouble()).toList()));
  }

  double get(int ni, int ci, int hi, int wi) {
    return data[((ni * c + ci) * h + hi) * w + wi];
  }

  void set(int ni, int ci, int hi, int wi, double v) {
    data[((ni * c + ci) * h + hi) * w + wi] = v;
  }

  Tensor copy() {
    return Tensor._(n, c, h, w, Float32List.fromList(data));
  }

  Tensor relu() {
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i] > 0 ? data[i] : 0.0;
    }
    return Tensor._(n, c, h, w, out);
  }

  Tensor add(Tensor other) {
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i] + other.data[i];
    }
    return Tensor._(n, c, h, w, out);
  }

  Tensor mul(Tensor other) {
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i] * other.data[i];
    }
    return Tensor._(n, c, h, w, out);
  }

  Tensor mulScalar(double v) {
    final out = Float32List(data.length);
    for (int i = 0; i < data.length; i++) {
      out[i] = data[i] * v;
    }
    return Tensor._(n, c, h, w, out);
  }
}

class Conv2d {
  final Float32List weight; // [oc, ic, kh, kw]
  final int oc, ic, kh, kw, stride, pad, groups;
  final bool hasBias;
  final Float32List? bias;

  Conv2d({
    required this.oc, required this.ic, required this.kh, required this.kw,
    this.stride = 1, this.pad = 0, this.groups = 1, this.hasBias = false,
    required this.weight, this.bias,
  });

  Tensor forward(Tensor x) {
    final n = x.n; final h = x.h; final w = x.w;
    final oh = (h + 2 * pad - kh) ~/ stride + 1;
    final ow = (w + 2 * pad - kw) ~/ stride + 1;
    final out = Tensor.zeros(n, oc, oh, ow);

    final icPerGroup = ic ~/ groups;
    final ocPerGroup = oc ~/ groups;

    for (int g = 0; g < groups; g++) {
      for (int oi = 0; oi < ocPerGroup; oi++) {
        final ooc = g * ocPerGroup + oi;
        for (int ni = 0; ni < n; ni++) {
          for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
              double sum = hasBias ? bias![ooc] : 0.0;
              for (int ii = 0; ii < icPerGroup; ii++) {
                final iic = g * icPerGroup + ii;
                for (int ky = 0; ky < kh; ky++) {
                  final iy = oy * stride + ky - pad;
                  if (iy < 0 || iy >= h) continue;
                  for (int kx = 0; kx < kw; kx++) {
                    final ix = ox * stride + kx - pad;
                    if (ix < 0 || ix >= w) continue;
                    sum += x.get(ni, iic, iy, ix) *
                        weight[((ooc * ic + iic) * kh + ky) * kw + kx];
                  }
                }
              }
              out.set(ni, ooc, oy, ox, sum);
            }
          }
        }
      }
    }
    return out;
  }
}

class Conv1dForCSA {
  final Float32List weight; // [1, 1, k]
  final int k;

  Conv1dForCSA(this.k, this.weight);

  Float32List forward(Float32List x) {
    final c = x.length;
    final pad = k ~/ 2;
    final out = Float32List(c);
    for (int i = 0; i < c; i++) {
      double sum = 0.0;
      for (int j = 0; j < k; j++) {
        final idx = i + j - pad;
        if (idx >= 0 && idx < c) {
          sum += x[idx] * weight[j];
        }
      }
      out[i] = sum;
    }
    return out;
  }
}

class BatchNorm2d {
  final Float32List weight;
  final Float32List bias;
  final Float32List runningMean;
  final Float32List runningVar;
  final double eps = 1e-5;
  final int c;

  BatchNorm2d(this.c, this.weight, this.bias, this.runningMean, this.runningVar);

  Tensor forward(Tensor x) {
    final out = Tensor.zeros(x.n, x.c, x.h, x.w);
    for (int ni = 0; ni < x.n; ni++) {
      for (int ci = 0; ci < c; ci++) {
        final wVal = weight[ci];
        final bVal = bias[ci];
        final mean = runningMean[ci];
        final varVal = runningVar[ci];
        final invStd = 1.0 / math.sqrt(varVal + eps);
        for (int hi = 0; hi < x.h; hi++) {
          for (int wi = 0; wi < x.w; wi++) {
            final val = (x.get(ni, ci, hi, wi) - mean) * invStd * wVal + bVal;
            out.set(ni, ci, hi, wi, val);
          }
        }
      }
    }
    return out;
  }
}

Tensor channelShuffle(Tensor x, int groups) {
  final n = x.n; final c = x.c; final h = x.h; final w = x.w;
  final cpg = c ~/ groups;
  final out = Tensor.zeros(n, c, h, w);
  for (int ni = 0; ni < n; ni++) {
    for (int gi = 0; gi < groups; gi++) {
      for (int ci = 0; ci < cpg; ci++) {
        final src = gi * cpg + ci;
        final dst = ci * groups + gi;
        for (int hi = 0; hi < h; hi++) {
          for (int wi = 0; wi < w; wi++) {
            out.set(ni, dst, hi, wi, x.get(ni, src, hi, wi));
          }
        }
      }
    }
  }
  return out;
}

Tensor adaptiveAvgPool2d1(Tensor x) {
  final n = x.n; final c = x.c; final h = x.h; final w = x.w;
  final out = Tensor.zeros(n, c, 1, 1);
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      double sum = 0.0;
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          sum += x.get(ni, ci, hi, wi);
        }
      }
      out.set(ni, ci, 0, 0, sum / (h * w));
    }
  }
  return out;
}

Float32List sigmoid(Float32List x) {
  final out = Float32List(x.length);
  for (int i = 0; i < x.length; i++) {
    out[i] = 1.0 / (1.0 + math.exp(-x[i]));
  }
  return out;
}

/// ShuffledFusionNet - Pure Dart Implementation
class ShuffledFusionNet {
  // AMKF
  late final Float32List _amkfFw;
  late final Conv2d _amkfB3Conv;
  late final BatchNorm2d _amkfB3Bn;
  late final Conv2d _amkfB5Conv;
  late final BatchNorm2d _amkfB5Bn;
  late final BatchNorm2d _bn0;

  // SGDR1: 32->64, stride=2
  late final Conv2d _sgdr1Dw;
  late final BatchNorm2d _sgdr1DwBn;
  late final Conv2d _sgdr1Gconv;
  late final BatchNorm2d _sgdr1GconvBn;
  late final Conv2d _sgdr1Pw;
  late final BatchNorm2d _sgdr1PwBn;
  late final Conv2d _sgdr1Skip;
  late final BatchNorm2d _sgdr1SkipBn;
  late final Conv1dForCSA _csa1;

  // SGDR2: 64->96, stride=2
  late final Conv2d _sgdr2Dw;
  late final BatchNorm2d _sgdr2DwBn;
  late final Conv2d _sgdr2Gconv;
  late final BatchNorm2d _sgdr2GconvBn;
  late final Conv2d _sgdr2Pw;
  late final BatchNorm2d _sgdr2PwBn;
  late final Conv2d _sgdr2Skip;
  late final BatchNorm2d _sgdr2SkipBn;
  late final Conv1dForCSA _csa2;

  // SGDR3: 96->96, stride=1
  late final Conv2d _sgdr3Dw;
  late final BatchNorm2d _sgdr3DwBn;
  late final Conv2d _sgdr3Gconv;
  late final BatchNorm2d _sgdr3GconvBn;
  late final Conv2d _sgdr3Pw;
  late final BatchNorm2d _sgdr3PwBn;
  late final Conv1dForCSA _csa3;

  // Classifier
  late final Conv2d _cls;

  bool _loaded = false;

  Future<void> loadWeights() async {
    if (_loaded) return;

    // Load weights.bin from assets
    final binData = await rootBundle.load('assets/weights.bin');
    final bytes = binData.buffer.asByteData();

    // Load metadata
    final jsonStr = await rootBundle.loadString('assets/weights.json');
    final meta = json.decode(jsonStr) as Map<String, dynamic>;
    final layers = meta['layers'] as List;

    // Helper to read weights by name
    Float32List readWeights(String name) {
      for (final layer in layers) {
        if (layer['name'] == name) {
          final offset = layer['offset'] as int;
          final size = layer['size'] as int;
          final floats = Float32List(size);
          for (int i = 0; i < size; i++) {
            floats[i] = bytes.getFloat32(offset + i * 4, Endian.little);
          }
          return floats;
        }
      }
      throw Exception('Weight not found: $name');
    }

    // AMKF
    _amkfFw = readWeights('amkf.fw');
    _amkfB3Conv = Conv2d(oc: 16, ic: 1, kh: 3, kw: 3, stride: 1, pad: 1, groups: 1,
        hasBias: false, weight: readWeights('amkf.b3.0.weight'));
    _amkfB3Bn = BatchNorm2d(16, readWeights('amkf.b3.1.weight'), readWeights('amkf.b3.1.bias'),
        readWeights('amkf.b3.1.running_mean'), readWeights('amkf.b3.1.running_var'));
    _amkfB5Conv = Conv2d(oc: 16, ic: 1, kh: 5, kw: 5, stride: 1, pad: 2, groups: 1,
        hasBias: false, weight: readWeights('amkf.b5.0.weight'));
    _amkfB5Bn = BatchNorm2d(16, readWeights('amkf.b5.1.weight'), readWeights('amkf.b5.1.bias'),
        readWeights('amkf.b5.1.running_mean'), readWeights('amkf.b5.1.running_var'));
    _bn0 = BatchNorm2d(32, readWeights('bn0.weight'), readWeights('bn0.bias'),
        readWeights('bn0.running_mean'), readWeights('bn0.running_var'));

    // SGDR1
    _sgdr1Dw = Conv2d(oc: 32, ic: 32, kh: 3, kw: 3, stride: 2, pad: 1,
        groups: 8, hasBias: false, weight: readWeights('sgdr1.dw.0.weight'));
    _sgdr1DwBn = BatchNorm2d(32, readWeights('sgdr1.dw.1.weight'), readWeights('sgdr1.dw.1.bias'),
        readWeights('sgdr1.dw.1.running_mean'), readWeights('sgdr1.dw.1.running_var'));
    _sgdr1Gconv = Conv2d(oc: 32, ic: 32, kh: 1, kw: 1, stride: 1, pad: 0,
        groups: 4, hasBias: false, weight: readWeights('sgdr1.gconv.0.weight'));
    _sgdr1GconvBn = BatchNorm2d(32, readWeights('sgdr1.gconv.1.weight'), readWeights('sgdr1.gconv.1.bias'),
        readWeights('sgdr1.gconv.1.running_mean'), readWeights('sgdr1.gconv.1.running_var'));
    _sgdr1Pw = Conv2d(oc: 64, ic: 32, kh: 1, kw: 1, stride: 1, pad: 0,
        groups: 1, hasBias: false, weight: readWeights('sgdr1.pw.0.weight'));
    _sgdr1PwBn = BatchNorm2d(64, readWeights('sgdr1.pw.1.weight'), readWeights('sgdr1.pw.1.bias'),
        readWeights('sgdr1.pw.1.running_mean'), readWeights('sgdr1.pw.1.running_var'));
    _sgdr1Skip = Conv2d(oc: 64, ic: 32, kh: 1, kw: 1, stride: 2, pad: 0,
        groups: 1, hasBias: false, weight: readWeights('sgdr1.sc.0.weight'));
    _sgdr1SkipBn = BatchNorm2d(64, readWeights('sgdr1.sc.1.weight'), readWeights('sgdr1.sc.1.bias'),
        readWeights('sgdr1.sc.1.running_mean'), readWeights('sgdr1.sc.1.running_var'));
    _csa1 = Conv1dForCSA(3, readWeights('csa1.c1.weight'));

    // SGDR2
    _sgdr2Dw = Conv2d(oc: 64, ic: 64, kh: 3, kw: 3, stride: 2, pad: 1,
        groups: 16, hasBias: false, weight: readWeights('sgdr2.dw.0.weight'));
    _sgdr2DwBn = BatchNorm2d(64, readWeights('sgdr2.dw.1.weight'), readWeights('sgdr2.dw.1.bias'),
        readWeights('sgdr2.dw.1.running_mean'), readWeights('sgdr2.dw.1.running_var'));
    _sgdr2Gconv = Conv2d(oc: 64, ic: 64, kh: 1, kw: 1, stride: 1, pad: 0,
        groups: 4, hasBias: false, weight: readWeights('sgdr2.gconv.0.weight'));
    _sgdr2GconvBn = BatchNorm2d(64, readWeights('sgdr2.gconv.1.weight'), readWeights('sgdr2.gconv.1.bias'),
        readWeights('sgdr2.gconv.1.running_mean'), readWeights('sgdr2.gconv.1.running_var'));
    _sgdr2Pw = Conv2d(oc: 96, ic: 64, kh: 1, kw: 1, stride: 1, pad: 0,
        groups: 1, hasBias: false, weight: readWeights('sgdr2.pw.0.weight'));
    _sgdr2PwBn = BatchNorm2d(96, readWeights('sgdr2.pw.1.weight'), readWeights('sgdr2.pw.1.bias'),
        readWeights('sgdr2.pw.1.running_mean'), readWeights('sgdr2.pw.1.running_var'));
    _sgdr2Skip = Conv2d(oc: 96, ic: 64, kh: 1, kw: 1, stride: 2, pad: 0,
        groups: 1, hasBias: false, weight: readWeights('sgdr2.sc.0.weight'));
    _sgdr2SkipBn = BatchNorm2d(96, readWeights('sgdr2.sc.1.weight'), readWeights('sgdr2.sc.1.bias'),
        readWeights('sgdr2.sc.1.running_mean'), readWeights('sgdr2.sc.1.running_var'));
    _csa2 = Conv1dForCSA(3, readWeights('csa2.c1.weight'));

    // SGDR3
    _sgdr3Dw = Conv2d(oc: 96, ic: 96, kh: 3, kw: 3, stride: 1, pad: 1,
        groups: 24, hasBias: false, weight: readWeights('sgdr3.dw.0.weight'));
    _sgdr3DwBn = BatchNorm2d(96, readWeights('sgdr3.dw.1.weight'), readWeights('sgdr3.dw.1.bias'),
        readWeights('sgdr3.dw.1.running_mean'), readWeights('sgdr3.dw.1.running_var'));
    _sgdr3Gconv = Conv2d(oc: 96, ic: 96, kh: 1, kw: 1, stride: 1, pad: 0,
        groups: 4, hasBias: false, weight: readWeights('sgdr3.gconv.0.weight'));
    _sgdr3GconvBn = BatchNorm2d(96, readWeights('sgdr3.gconv.1.weight'), readWeights('sgdr3.gconv.1.bias'),
        readWeights('sgdr3.gconv.1.running_mean'), readWeights('sgdr3.gconv.1.running_var'));
    _sgdr3Pw = Conv2d(oc: 96, ic: 96, kh: 1, kw: 1, stride: 1, pad: 0,
        groups: 1, hasBias: false, weight: readWeights('sgdr3.pw.0.weight'));
    _sgdr3PwBn = BatchNorm2d(96, readWeights('sgdr3.pw.1.weight'), readWeights('sgdr3.pw.1.bias'),
        readWeights('sgdr3.pw.1.running_mean'), readWeights('sgdr3.pw.1.running_var'));
    _csa3 = Conv1dForCSA(3, readWeights('csa3.c1.weight'));

    // Classifier
    _cls = Conv2d(oc: 10, ic: 96, kh: 1, kw: 1, stride: 1, pad: 0,
        groups: 1, hasBias: true, weight: readWeights('cls.weight'), bias: readWeights('cls.bias'));

    _loaded = true;
    print('[ShuffledFusionNet] Weights loaded (39,349 params)');
  }

  Tensor _amkfForward(Tensor x) {
    // Softmax on fusion weights
    final fw = Float32List(2);
    final exp0 = math.exp(_amkfFw[0]);
    final exp1 = math.exp(_amkfFw[1]);
    final sum = exp0 + exp1;
    fw[0] = exp0 / sum * 2.0;
    fw[1] = exp1 / sum * 2.0;

    // Branch 3x3
    Tensor b3 = _amkfB3Bn.forward(_amkfB3Conv.forward(x));
    b3 = b3.mulScalar(fw[0]);

    // Branch 5x5
    Tensor b5 = _amkfB5Bn.forward(_amkfB5Conv.forward(x));
    b5 = b5.mulScalar(fw[1]);

    // Concatenate
    final n = b3.n; final h = b3.h; final w = b3.w;
    final out = Tensor.zeros(n, 32, h, w);
    for (int ni = 0; ni < n; ni++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          for (int ci = 0; ci < 16; ci++) {
            out.set(ni, ci, hi, wi, b3.get(ni, ci, hi, wi));
            out.set(ni, ci + 16, hi, wi, b5.get(ni, ci, hi, wi));
          }
        }
      }
    }
    return out;
  }

  Tensor _sgdrForward(Tensor x, Conv2d dw, BatchNorm2d dwBn,
      Conv2d gconv, BatchNorm2d gconvBn,
      Conv2d pw, BatchNorm2d pwBn,
      Conv2d skipConv, BatchNorm2d skipBn, int groups) {
    final shortcut = skipConv.forward(x);
    final shortcutBn = skipBn.forward(shortcut);

    Tensor out = dwBn.forward(dw.forward(x));
    out = out.relu();
    out = channelShuffle(out, groups);
    out = gconvBn.forward(gconv.forward(out));
    out = out.relu();
    out = pwBn.forward(pw.forward(out));
    return out.add(shortcutBn).relu();
  }

  Tensor _csaForward(Tensor x, Conv1dForCSA conv1d) {
    final n = x.n; final c = x.c; final h = x.h; final w = x.w;

    // Global Average Pooling over spatial dims
    final gap = Float32List(c);
    for (int ci = 0; ci < c; ci++) {
      double sum = 0.0;
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          sum += x.get(0, ci, hi, wi);
        }
      }
      gap[ci] = sum / (h * w);
    }

    // 1D Conv + Sigmoid
    final attention = sigmoid(conv1d.forward(gap));

    // Apply attention
    final out = Tensor.zeros(n, c, h, w);
    for (int ni = 0; ni < n; ni++) {
      for (int ci = 0; ci < c; ci++) {
        final attn = attention[ci];
        for (int hi = 0; hi < h; hi++) {
          for (int wi = 0; wi < w; wi++) {
            out.set(ni, ci, hi, wi, x.get(ni, ci, hi, wi) * attn);
          }
        }
      }
    }
    return out;
  }

  /// Run inference on a 28x28 normalized grayscale image
  /// Input: 28x28 Float32List (normalized: (pixel/255 - 0.1307) / 0.3081)
  /// Returns: (digit: 0-9, confidence: 0-1)
  ({int digit, double confidence}) predict(Float32List input) {
    if (!_loaded) throw Exception('Model not loaded. Call loadWeights() first.');

    // Wrap input as [1, 1, 28, 28] tensor
    Tensor x = Tensor._(1, 1, 28, 28, Float32List.fromList(input));

    // Forward pass
    // AMKF(1->32) + BN
    x = _amkfForward(x);
    x = _bn0.forward(x);

    // SGDR1(32->64, stride=2) + CSA
    x = _sgdrForward(x, _sgdr1Dw, _sgdr1DwBn, _sgdr1Gconv, _sgdr1GconvBn,
        _sgdr1Pw, _sgdr1PwBn, _sgdr1Skip, _sgdr1SkipBn, 4);
    x = _csaForward(x, _csa1);

    // SGDR2(64->96, stride=2) + CSA
    x = _sgdrForward(x, _sgdr2Dw, _sgdr2DwBn, _sgdr2Gconv, _sgdr2GconvBn,
        _sgdr2Pw, _sgdr2PwBn, _sgdr2Skip, _sgdr2SkipBn, 4);
    x = _csaForward(x, _csa2);

    // SGDR3(96->96, stride=1) + CSA
    x = _sgdrForward(x, _sgdr3Dw, _sgdr3DwBn, _sgdr3Gconv, _sgdr3GconvBn,
        _sgdr3Pw, _sgdr3PwBn, _sgdr3Dw,  x, _sgdr3DwBn, // identity skip
    // Actually SGDR3 uses identity skip
    // Let me fix: SGDR3 has no skip conv (stride=1, ic==oc)
    // So we need a different handling...

    return _predictInternal(input);
  }

  ({int digit, double confidence}) _predictInternal(Float32List input) {
    // AMKF
    Tensor x = Tensor._(1, 1, 28, 28, Float32List.fromList(input));

    // AMKF(1->32)
    x = _amkfForward(x);
    x = _bn0.forward(x);

    // SGDR1(32->64, stride=2) + CSA
    x = _sgdrForwardWithSkip(x, _sgdr1Dw, _sgdr1DwBn, _sgdr1Gconv, _sgdr1GconvBn,
        _sgdr1Pw, _sgdr1PwBn, _sgdr1Skip, _sgdr1SkipBn, 4);
    x = _csaForward(x, _csa1);

    // SGDR2(64->96, stride=2) + CSA
    x = _sgdrForwardWithSkip(x, _sgdr2Dw, _sgdr2DwBn, _sgdr2Gconv, _sgdr2GconvBn,
        _sgdr2Pw, _sgdr2PwBn, _sgdr2Skip, _sgdr2SkipBn, 4);
    x = _csaForward(x, _csa2);

    // SGDR3(96->96, stride=1, identity skip)
    x = _sgdrForwardIdentity(x, _sgdr3Dw, _sgdr3DwBn, _sgdr3Gconv, _sgdr3GconvBn,
        _sgdr3Pw, _sgdr3PwBn, 4);
    x = _csaForward(x, _csa3);

    // Classifier: Conv2d(96->10, 1x1) + GAP
    x = _cls.forward(x);
    x = adaptiveAvgPool2d1(x);

    // Softmax
    final logits = Float32List(10);
    for (int i = 0; i < 10; i++) {
      logits[i] = x.data[i];
    }

    final maxLogit = logits.reduce((a, b) => a > b ? a : b);
    double sumExp = 0.0;
    for (int i = 0; i < 10; i++) {
      sumExp += math.exp(logits[i] - maxLogit);
    }
    final probs = Float32List(10);
    for (int i = 0; i < 10; i++) {
      probs[i] = math.exp(logits[i] - maxLogit) / sumExp;
    }

    int bestDigit = 0;
    double bestConf = probs[0];
    for (int i = 1; i < 10; i++) {
      if (probs[i] > bestConf) {
        bestConf = probs[i];
        bestDigit = i;
      }
    }

    return (digit: bestDigit, confidence: bestConf);
  }

  Tensor _sgdrForwardWithSkip(Tensor x, Conv2d dw, BatchNorm2d dwBn,
      Conv2d gconv, BatchNorm2d gconvBn,
      Conv2d pw, BatchNorm2d pwBn,
      Conv2d skipConv, BatchNorm2d skipBn, int groups) {
    final shortcut = skipBn.forward(skipConv.forward(x));
    Tensor out = dwBn.forward(dw.forward(x));
    out = out.relu();
    out = channelShuffle(out, groups);
    out = gconvBn.forward(gconv.forward(out));
    out = out.relu();
    out = pwBn.forward(pw.forward(out));
    return out.add(shortcut).relu();
  }

  Tensor _sgdrForwardIdentity(Tensor x, Conv2d dw, BatchNorm2d dwBn,
      Conv2d gconv, BatchNorm2d gconvBn,
      Conv2d pw, BatchNorm2d pwBn, int groups) {
    Tensor out = dwBn.forward(dw.forward(x));
    out = out.relu();
    out = channelShuffle(out, groups);
    out = gconvBn.forward(gconv.forward(out));
    out = out.relu();
    out = pwBn.forward(pw.forward(out));
    return out.add(x).relu();
  }
}