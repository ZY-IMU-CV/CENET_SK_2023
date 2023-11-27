CENet运行步骤：
1.按照官方文档配置mmsegmentation环境，官方地址https://mmsegmentation.readthedocs.io/en/latest/
2.将code_cenet\model文件夹下的CENet.py中的代码复制并粘贴直接覆盖掉\mmseg\models\decode_heads中的sep_aspp_head.py的源码
3.将code_cenet\configs中的对应的配置文件拷贝到mmsegmentation\configs\deeplabv3plus文件夹下
4.按照官方文档说明的步骤执行对应的配置文件即可训练模型

注：文件夹code_cenet\mmsegmentation项目无法克隆后直接运行，仅为配置文件的拷贝位置提供参考，实际操作需按步骤1参考官方文档进行环境配置。