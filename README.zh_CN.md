[English](/README.md) | [简体中文](/README.zh_CN.md)

<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="img/PEPL-logo.png" alt="Project logo"></a>
</p>
<h2 align="center"><i>Part-of-speech Enhanced Prompt Learning</i>  for clickbait detection</h2>

<div align="center">

  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
  ![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
  [![Python](https://img.shields.io/badge/python-3.10.8-brightgreen)](https://www.python.org/downloads/release/python-3108/)
  [![Pytorch](https://img.shields.io/badge/pytorch-1.13.0-orange)](https://pytorch.org/get-started/previous-versions/#v1130)
  ![Last Commit](https://img.shields.io/github/last-commit/WeSeewy/Chinese-Clickbait)

</div>

---



## 📝 目录
- [介绍](#about)
- [准备工作](#getting_started)
- [使用指南](#usage)
- [贡献者](#authors)
- [改进项目](#contributing)

## 🧐 介绍 <a name = "about"></a>
本项目实现了一种面向中文点击诱饵（又名“标题党”）新闻的检测方法，其能够判断输入的中文文本是否属于点击诱饵标题。点击诱饵与非点击诱饵的示例如下表所示：

|  示例   | 分类  |
|  ----  | ----  |
| 当女生问“你在干啥”，这句回答100%勾起她的兴趣  | 点击诱饵 |
| 北京发布做好复工复产疫情防控常态化工作通告  | 非点击诱饵 |

本项目的主要想法撰写于论文*Detecting Clickbait in Chinese Social Media by Prompt Learning*，该文章已被CSCWD'23接收。

## 🏁 准备工作 <a name = "getting_started"></a>
为使本项目能正常运行，首先需要进行准备工作。

### 框架依赖
本项目运行时主要依赖如下：

- [Pytorch](https://pytorch.org/) - 深度学习框架，请参考[指南](https://pytorch.org/get-started/previous-versions/#v1130)进行安装，版本为1.13.0
- [LTP](https://github.com/HIT-SCIR/ltp) - 中文文本处理工具，请参考[指南](https://github.com/HIT-SCIR/ltp)进行安装，版本为4.2.11
- [Transformers](https://github.com/huggingface/transformers) - 大模型管理框架，请参考[指南](https://huggingface.co/docs/transformers/installation)进行安装，版本为4.24.0
- [OpenPrompt](https://github.com/thunlp/OpenPrompt) - 开源提示学习框架，请参考[指南](https://github.com/thunlp/OpenPrompt#installation)进行安装，版本为1.0.1

此外，还有其他的依赖包需要进行安装，如果您使用的python版本为3.10.8，则可以直接执行如下命令进行安装：


```
pip install -r requirements.txt
```

如果您使用的是其他python版本，请修改requirements.txt文件中的版本以使其兼容。

### 数据集

本项目选择[WCD](https://github.com/natsusaikou/WeChat-Clickbait)数据集进行训练和测试，请下载其中的[all_labeled.csv](https://github.com/natsusaikou/WeChat-Clickbait/blob/master/data/all_labeled.csv)文件并将其放置于 **/data** 路径。如您必须要修改数据集位置，请修改 **main.py** 文件中的DatasetPath变量。

如果您需要使用其他数据集，请修改 **preprocess.py** 与 **helper/loader.py** 文件，本项目仅需中文文本字段作为输入。


## 🎈 使用指南 <a name="usage"></a>
### 训练

在完成上述准备工作后，可以使用如下指令进行训练：

```
python main.py -s 0.01 0.5
```

该训练指令-s后两参数的含义是将1%的数据集作为训练集，将50%的数据集作为测试集。

另外也可以使用如下指令在极端小样本场景中训练模型：

```
python main.py -f 16
```

该训练指令-f后参数的含义是仅有16个点击诱饵和16个非点击诱饵样本用于训练。

此外，指令执行还支持其他设置，具体情况如下：

- -bs：设置训练中的batch size
- -lr：设置训练学习率
- -ep：设置训练epoch
- -m：设置方法模式，仅支持*pepl*和*base*，其中*pepl*为本项目方法，*base*则为基线方法
- -plm：设置方法所基于的大模型架构，支持*bert*、*roberta*、*ernie*、*Erlangshen*

如下为设置较为全面的训练指令示例：

```
python main.py -s 0.01 0.5 -bs 8 -lr 5e-5 -ep 3 -m pepl -plm bert
```

### 预测

在经过上述训练后，模型将会被存储于 **/checkpoints** 文件夹内。执行如下指令即可对数据进行预测，其中 **/data** 文件夹内的items.txt与news.txt文件即为类似的待预测样本。

```
python use.py -p .\checkpoints\base_bert_fc_2.pt -m base -plm bert
```

参数含义如下所示：

- -p：设置模型存储位置，例如本例表示的base_bert_fc_2.pt模型
- -m：设置方法模式，仅支持*pepl*和*base*
- -plm：设置方法所基于的大模型架构，支持*bert*、*roberta*、*ernie*、*Erlangshen*

**注意：使用时-m和-plm应与模型相匹配，即与训练阶段相同**

## ✍️ 贡献者 <a name = "authors"></a>
- [@WeSeewy](https://github.com/WeSeewy) - Idea & Develop
- [@caomingpei](https://github.com/caomingpei) - Test & Doc

其他贡献者请查看此[列表](https://github.com/WeSeewy/Chinese-Clickbait/contributors).

## ⛏️ 改进项目 <a name = "contributing"></a>
如果您发现项目中存在的问题，请提交issue。

如果您想向本项目贡献代码，请fork并创建新的pull request。

Git提交风格请遵循如下约定：

```
[!TYPE:] message
```

[!TYPE:] 包括如下类型约定：

- !F: 实现新功能
- !B: 修复漏洞
- !D: 更新文档
- !S: 更改代码格式
- !R: 重构代码，不改变代码功能或性能
- !O: 代码优化
- !T: 增加测试
- !C: 更新依赖或其他
- !A: 存档相关文件

以下为git提交示例：

```
!D: configuring the git style
```

其中 [!D:] 表示是对文档内容进行更新，commit的主要内容在于设置了git格式。

