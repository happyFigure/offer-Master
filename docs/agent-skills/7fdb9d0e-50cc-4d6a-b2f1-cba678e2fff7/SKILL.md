---
name: wechat-article-content-fetch
description: 当用户提供微信公众号文章链接、公众号招聘文章或希望从公众号内容中抽取秋招开放信息时使用；读取公开文章正文、图片线索和视频关键帧，不自动投递。
source_types:
  - wechat_article
  - wechat_account
required_tools:
  - weixin-articles-mcp.read_article
allowed_tools:
  - weixin-articles.read_article
  - weixin-articles-mcp.read_article
  - read_article
  - ocr.extract_text
  - memory_search
  - memory_get
ask_tools:
  - browser.open
  - mcp.visible_page_read
disallowed_tools:
  - submit_application
compatibility:
  - offermaster
  - claude-code-style
license: MIT
---

# WeChat Article Content Fetch

这个 Skill 用来告诉 OfferMaster Agent：当用户给出微信公众号文章链接，或者希望从公众号招聘文章里提取秋招开放信息时，优先调用公众号文章读取 MCP，再把正文、图片和视频关键帧交给后续抽取流程。

## 外部来源

- 外部仓库：`vendor/external-skills/weixin-articles-mcp`
- 仓库用途：读取公开可访问的 `mp.weixin.qq.com/s/...` 文章 URL。
- 主要工具：`read_article(url)`，返回文章元数据、Markdown 正文、图片内容块和部分视频关键帧。
- 当前边界：它读取单篇公开文章，不负责按公众号账号自动拉取历史文章列表。

## 输入

- 微信公众号文章 URL，例如 `https://mp.weixin.qq.com/s/...`。
- 用户粘贴的公众号招聘文章正文。
- 后续扩展时，可以接收公众号账号线索，但账号历史文章同步需要单独的可见页面工具或公众号文章列表工具。

## 输出

- 文章标题、公众号名称、发布时间和原始 URL。
- 文章正文 Markdown。
- 图片中的招聘文字线索，必要时交给 `ocr.extract_text`。
- 从正文中识别出的公司秋招开放信号、岗位方向、届别、投递入口和可信度。

## 工作流程

1. 判断 URL 是否属于 `mp.weixin.qq.com`。
2. 调用 `weixin-articles.read_article` 获取文章正文和图片内容。
3. 如果招聘信息在图片里，调用 OCR 抽取图片文字。
4. 用模型只做结构化抽取，不把抽取结果直接当成正式岗位。
5. 把原始文章保存为 raw lead，把公司开放信号保存为 recruiting signal。
6. 真正投递前，再走企业官网或招聘官网验证。

## 权限边界

- 允许读取公开文章和做 OCR。
- 打开浏览器或读取用户当前可见页面时必须走确认。
- 禁止自动投递、自动提交申请或绕过登录、验证码、反爬机制。
- 不保存公众号登录态，不使用用户 Cookie，不做高频抓取。

## 失败处理

- 如果文章无法抓取，返回结构化错误，提示用户改用可见页面或手动粘贴正文兜底。
- 如果正文为空但图片存在，优先尝试 OCR。
- 如果只识别出“某公司开放秋招”但没有具体 JD，记录为招聘开放信号，后续再补全岗位。
