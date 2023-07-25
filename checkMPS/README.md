当前目录代码用于测试GPU是否真的支持MPS。
# 前提
已安装NVIDIA驱动，并配好了环境变量。
**注意：**普通用户和root用户的`bashrc`中都要配置cuda环境配置。
# 实现
## 文件功能
- `vectoradd.cu`: 对两个很大的数组做加法，因此单个`vectoradd`程序运行时间就会很长（cuda11.7 2080s 执行时间约5s）
- `test-mps.sh`: 自动编译、测试代码
## 测试原理
1. 编译cuda程序`vectoradd`
2. 执行一次`vectoradd`，获得执行时间`basetime`
3. 在**不开启MPS**的环境中，并行执行两个`vectoradd`，获得执行时间`execution_time_a`。理论上，$execution_time_a≈2*basetime$
4. 在**开启MPS**的环境中，并行执行两个`vectoradd`，获得执行时间`execution_time_b`。理论上，$execution_time_b≈basetime$ 且 $execution_time_a >> execution_time_b$
5. 因此，假设若`execution_time_a`大于`execution_time_b` 2s以上，即认为当前GPU支持MPS
# 使用
克隆项目到本地，在当前文件夹中打开终端，输入以下命令：
```shell
# 进入root用户的终端
sudo bash -c bash
chmod +x test-mps.sh
./test-mps.sh
```