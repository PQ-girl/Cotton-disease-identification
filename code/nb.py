# paddle模型导出安卓端可以运行的nb文件
# paddlehub模型库网址：https://www.paddlepaddle.org.cn/hublist

# 引用Paddlelite预测库
from paddlelite.lite import *


def convert():
    # 1. 创建opt实例
    opt = Opt()
    # 2. 指定输入模型地址
    opt.set_model_file("./pd_model1/1.pdmodel")
    opt.set_param_file("./pd_model1/1.pdiparams")
    # opt.set_model_dir(r"D:\project\nb\model\ace2p\ace2p_human_parsing")

    # 3. 指定转化类型： arm、x86、opencl、npu
    opt.set_valid_places("arm")
    # 4. 指定模型转化类型： naive_buffer、protobuf
    opt.set_model_type("naive_buffer")
    # 4. 输出模型地址
    opt.set_optimize_out("model224")
    # 5. 执行模型优化
    opt.run()
    # opt.run_optimize("", "", "model.pdparams", "arm,npu", "deepl3p_opt")


if __name__ == '__main__':
    print('导出模型开始....')
    convert()
    print('导出结束.....')
