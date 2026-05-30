# ShuffledFusionNet App - Windows开发 → iPhone安装

## 方案：Flutter + Codemagic 云端编译

Windows开发Flutter代码 → GitHub → Codemagic云端编译iOS → TestFlight安装到iPhone

## 第一步：安装 Flutter (Windows)

```powershell
# 1. 下载 Flutter SDK
# https://docs.flutter.dev/get-started/install/windows

# 2. 解压到 D:\flutter

# 3. 添加环境变量 PATH += D:\flutter\bin

# 4. 验证
flutter doctor
```

## 第二步：运行测试 (Windows上预览)

```bash
cd digit_app
flutter pub get
flutter run -d windows
```

## 第三步：推送到 GitHub

```bash
git init
git add .
git commit -m "ShuffledFusionNet app"
# 在GitHub创建仓库 digit-app
git remote add origin https://github.com/YOUR_USER/digit-app.git
git push -u origin main
```

## 第四步：Codemagic 云端编译 iOS

1. 注册 https://codemagic.io (免费)
2. 连接 GitHub 仓库
3. 添加 `codemagic.yaml` 配置
4. 上传 Apple 开发者证书
5. 点击 Build → 自动编译 iOS IPA
6. 通过 TestFlight 安装到 iPhone

### codemagic.yaml (放在项目根目录)
```yaml
workflows:
  ios-build:
    name: iOS Build
    environment:
      flutter: stable
      xcode: latest
      cocoapods: default
    scripts:
      - flutter pub get
      - flutter build ios --release --no-codesign
    artifacts:
      - build/ios/ipa/*.ipa
```

## 第五步：iPhone 上安装

1. iPhone 下载 TestFlight (App Store)
2. Codemagic 构建完成后收到邮件
3. 点击链接 → TestFlight 安装
4. 拍照 → 识别！

## 项目结构

```
digit_app/
├── lib/main.dart              ← Flutter App 主代码
├── assets/
│   └── shuffled_fusion_net.pt ← 模型 (39K, 99.07%)
├── pubspec.yaml               ← 依赖配置
├── ios/Runner/Info.plist      ← 相机权限
└── codemagic.yaml             ← 云端编译配置
```
