## MNN Tutorial
- 缘起：由于MNN的文档并不直观，对于想要尝试MNN推理引擎的小伙伴来说，门槛有些高。（起码对于我来说是这样的），因此有了这个仓库。
- 整理使用一个推理引擎过程中，最基本使用的代码段，保证可以快速跑通样例示例程序。

#### MNN官方文档
- [最新文档](https://mnn-docs.readthedocs.io/en/latest/index.html)
- [语雀文档](https://www.yuque.com/mnn/cn)

#### [MNN官网](https://www.mnn.zone/)

#### ONNX模型转换为`mnn`格式
```bash
$ pip install MNN
$ MNNConvert -f ONNX --modelFile XXX.onnx --MNNModel XXX.mnn --bizCode biz
```

#### 模型推理
- [Session API](test_mnn_session_api.py)
- [Expr API](test_mnn_expr_api.py)

#### 动态输入推理
- TODO

#### CPU/GPU设备支持
- CPU可以通过config配置实现
  ```bash
    # 创建interpreter
    interpreter = MNN.Interpreter("models/mobilenet_v2-b0353104.mnn")

    # 创建session
    config = {}
    config['precision'] = 'high'
    config['backend'] = 'CPU'  # CUDA
    config['thread'] = 4
    session = interpreter.createSession(config)
  ```
- GPU设备
  - TODO

#### 参考资料
- [MNN框架C++和Python API Demo](https://blog.csdn.net/wl1710582732/article/details/107731147)
