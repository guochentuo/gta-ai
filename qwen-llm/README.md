# 大语言模型

本项目负责Qwen3.8-27B-FP8推理、vLLM运行时、专用Router、业务知识检索、CPU会话摘要
和可选LoRA训练，不执行ASR、OCR或媒体任务编排。

运行服务：`gta-ai-vllm.service`和`gta-ai-router.service`。Router只管理本项目的27B
GPU请求，不代理其他模型。

## CPU会话摘要

`gta-ai-qwen-summarizer.service`使用CPU运行`Qwen3-4B-Instruct-2507-Q4_K_M`，仅监听
`127.0.0.1:7107`，不通过Nginx或防火墙对外开放。调用方继续请求27B Router，并使用
`X-GTA-Session-ID`传递稳定会话ID。

Router根据`X-GTA-Session-ID`从Redis读取长期摘要和尚未压缩的最近轮次，再与当前问题一起
生成回答。回答完成后先把本轮问答写入Redis，然后异步调用CPU摘要模型更新长期记忆；因此
用户快速发送下一条时也不会等待摘要或丢失上一轮。摘要失败时继续使用旧摘要和未压缩消息，
不影响27B正常回答。Redis集群节点和密码直接配置在本项目的`config/router.env`中，
不读取Java项目或Nacos配置。
SQLite实现仅用于单元测试和显式回退，不再作为正式多节点会话存储。

传给27B的会话记忆采用固定Token预算。超过预算时优先保留压缩后的长期事实、与当前问题
相关的历史以及最近两轮，旧寒暄和无关内容不会进入本轮推理。预算由
`GTA_AI_SUMMARIZATION_MAX_CONTEXT_TOKENS`配置，避免会话越长Prefill越慢。

正式前端只需要发送稳定的`X-GTA-Session-ID`、每次唯一的`X-GTA-Request-ID`和当前问题；
完整聊天仍应由业务数据库长期保存，Redis只保存模型推理所需的热会话记忆并设置过期时间。

CPU 4B只承担后台会话摘要，不处理用户聊天问答。首屏欢迎模板本地化和所有用户问题统一交给
27B；本地化直接调用vLLM本机内部端口，不经过Router，避免递归请求。

## 首屏欢迎模板本地化

ES索引`gta_ai_welcome_template_v1`只保存中文母版。Router优先读取调用方传入的
`X-GTA-Locale`，并结合`X-GTA-Country`选择当地语言和表达习惯。中文直接返回母版；其他
语言先读取Redis缓存，未命中时由本机27B一次性本地化正文、界面文字和推荐问题，再将
结果缓存30天。缓存键包含模板ID、模板版本、语言和国家，中文母版版本更新后会自动生成新
缓存，不需要删除旧键。

Router启动后会在后台顺序预热英语、繁体中文、日语、韩语、德语、法语、西班牙语和泰语，
不阻塞端口启动；其他语言在首次真实访问时按需生成。内部图片标记、评价标记、URL和联系
账号不会交给模型改写。

## 日志

27B Router和vLLM使用相同的精简日志规则：

- 控制台只显示启动、停止和错误，不打印访问日志及模型逐请求细节；
- 启动时显示Router路由、模型、并发、GPU型号、显存和模型加载结果；
- 每5秒显示模型连接、GPU状态、输入及输出`token/s`、运行数和等待数；
- 模型首次加载成功后自动执行一次短测速，打印生成token数、耗时和平均`token/s`；
- 每个推理请求返回第一段数据时，控制台只打印首次回复耗时；请求ID仅保留在文件和Kafka；
- 每次回复结束后打印输入、输出、总token数及总回复时长；
- Router运行及错误日志分别写入`/opt/gta-ai/qwen-llm/logs/router.log`和
  `/opt/gta-ai/qwen-llm/logs/error.log`；vLLM写入`vllm.log`和`vllm.error.log`。
  日志达到20 MB后自动滚动，保留10份；
- 错误同时写入各自的`.error.log`，Python异常保留完整堆栈；
- 完整日志异步发送到统一日志topic `gta.video.worker.logs`，Kafka不可用时不阻塞推理，
  本地文件继续记录；
- 不记录提示词、请求正文和模型回答，只记录请求ID、耗时、状态及输入字节数。

日志目录、滚动大小、保留份数、Kafka地址和topic均由`config/router.env`及
`config/vllm.env`中的`GTA_AI_LOG_*`配置控制。

Debugger会覆盖日志目录为源码中的`qwen-llm/logs/`，便于直接在VS Code查看；运行日志
已被Git忽略，不会进入代码提交。

Router的GPU准入lease及waiter每5秒续期。进程异常退出或Debugger重启后，遗留状态最多
15秒自动失效，不能长期阻塞后续推理请求。
