---
title: "从工具调用到自主任务：Agent 学习笔记"
description: "从语言模型与工具的边界出发，理解 Agent 的目标、行动、观察和反馈闭环。"
tags: [Agent]
category: AGENTS
icon: steps
math: false
---

语言模型擅长根据上下文生成下一步内容，但真实任务通常不会停在“生成一段文字”。它需要查资料、运行代码、读写文件，并根据结果调整下一步。

当这些能力被组织成一个持续反馈的过程，我就开始从“聊天模型”走向“Agent”。

## Agent 多出来的是什么

一个只负责回答的模型，输入问题后输出答案。Agent 则通常处在一个循环里：

1. 理解目标与约束；
2. 决定下一步行动；
3. 调用工具或改变环境；
4. 读取行动结果；
5. 判断是否继续、修正或停止。

![Agent 的目标、行动、观察与调整闭环]({{ '/assets/images/posts/agent-loop.svg' | relative_url }})
<p class="figure-caption">关键不只是会调用工具，而是能利用调用结果调整后续行动。</p>

因此，工具调用只是其中一步。真正让系统具有“任务感”的，是行动之后还有观察，观察之后还能更新计划。

## 工具到底是什么

对模型来说，工具不是软件界面上的按钮，而是一份可调用的能力描述：名称、用途、参数和返回结果。

例如，一个搜索工具至少需要说明：

```json
{
  "name": "search",
  "description": "Search public documentation for a given query.",
  "parameters": {
    "query": "string"
  }
}
```

模型负责提出调用意图，运行环境负责真正执行。这个边界很重要：模型说“我要搜索”并不等于搜索已经发生，只有宿主程序解析请求、校验参数并调用搜索接口后，观察结果才存在。

这也意味着，工具设计不只是工程包装。模糊的说明、不稳定的参数格式和缺少上下文的返回值，都会直接增加模型决策的难度。

## ReAct：把推理与行动交错起来

ReAct 论文把推理轨迹和任务行动交错生成。可以把它简化成：

```text
Thought: 我需要先找到论文中的定义。
Action: search("RoFormer relative position")
Observation: 找到论文摘要与相关段落。
Thought: 现在可以比较绝对位置和相对位置。
Action: ...
```

推理帮助模型决定和修正行动，行动则让模型从外部环境获得新信息。二者结合后，模型不必在第一次看到问题时就写出完整且不可更改的答案。

不过，交错的文字轨迹不等于可靠执行。真正的系统仍需要解析、验证、错误处理和停止条件。

## 一个自主任务的简化循环

下面只是说明结构的伪代码，不对应某个具体框架：

```python
state = {"goal": goal, "history": [], "steps": 0}

while not state["done"] and state["steps"] < MAX_STEPS:
    decision = model.decide(state)
    if decision.type == "finish":
        state["done"] = True
        break

    validate(decision)
    observation = tools.execute(decision)
    state["history"].append((decision, observation))
    state["steps"] += 1
```

这里有几件事与模型能力同样重要：参数校验、最大步数、工具异常、历史组织和最终停止。如果只保留 `model.decide()`，演示可能看起来聪明，但系统很难稳定运行。

## 规划、记忆与自主程度

规划不等于一次性写出一长串步骤。环境会变化，工具会失败，因此计划需要能被观察结果修订。

记忆也不等于把所有历史塞进上下文。短期状态需要保留当前任务相关信息，长期记忆则要考虑检索、过期和隐私。上下文越多，并不一定越容易做出正确决定。

“自主”也不是一个非黑即白的属性。只允许读文件与允许删除文件，是完全不同的权限；只能调用一次搜索与可以无限循环，也是不同的自主范围。更准确的描述是：**系统能在什么权限和预算内，连续完成多少步。**

## 我会优先加上哪些约束

如果要把一个 Agent 从演示变成可用工具，我会先考虑：

- 工具权限最小化，敏感操作需要确认；
- 写操作尽量可撤销，重复执行要避免造成重复副作用；
- 为步数、时间和费用设置预算；
- 保存行动与观察日志，便于定位错误；
- 对结果进行验证，不把模型自己的“完成了”当作唯一依据。

这些约束不是削弱智能，而是在给智能划定可靠的运行环境。

## 我的当前理解

Agent 不是“更会说话的模型”，而是模型、工具、状态和反馈共同组成的系统。模型决定下一步，工具改变外部状态，观察结果再回到下一次决策。

把这条闭环看清楚，很多听起来宏大的概念就能落到具体问题上：它能调用什么，如何知道结果，错了怎么办，什么时候必须停下来。

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
