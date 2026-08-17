---
name: xiaohongshu-content-fetch
description: 当用户提供小红书笔记链接、账号、关键词或可见页面内容，并希望抽取秋招开放公司、岗位线索或图片中的招聘信息时使用；默认走用户可见页面和授权边界，不后台硬爬。
source_types:
  - xiaohongshu_note
required_tools:
  - xiaohongshu-mcp.search_feeds
  - xiaohongshu-mcp.get_feed_detail
allowed_tools:
  - xiaohongshu-mcp.check_login_status
  - xiaohongshu-mcp.search_feeds
  - xiaohongshu-mcp.get_feed_detail
  - xiaohongshu-mcp.user_profile
  - xhscrawl.search
  - xhscrawl.note_detail
  - xhscrawl.user_notes
  - ocr.extract_text
  - memory_search
  - memory_get
ask_tools:
  - xiaohongshu-mcp.get_login_qrcode
  - browser.open
  - mcp.visible_page_read
disallowed_tools:
  - xiaohongshu-mcp.publish_content
  - xiaohongshu-mcp.publish_with_video
  - xiaohongshu-mcp.post_comment_to_feed
  - xiaohongshu-mcp.reply_comment_in_feed
  - xiaohongshu-mcp.like_feed
  - xiaohongshu-mcp.favorite_feed
  - submit_application
compatibility:
  - offermaster
  - claude-code-style
license: Apache-2.0 / external license pending for xhscrawl
---

# Xiaohongshu Content Fetch

这个 Skill 用来告诉 OfferMaster Agent：当用户希望从小红书笔记、账号主页、关键词搜索结果或用户可见页面里提取秋招信息时，优先使用小红书 MCP 的读取类工具，并把图片文字交给 OCR，最终只沉淀招聘线索，不自动投递。

## 外部来源

- 外部仓库一：`vendor/external-skills/xiaohongshu-mcp-complete`
- 主要 MCP 工具：`search_feeds`、`get_feed_detail`、`user_profile`。
- 外部仓库二：`vendor/external-skills/xhscrawl`
- xhscrawl 参考能力：笔记搜索、笔记详情、用户搜索、用户发布笔记列表。

## 输入

- 小红书笔记链接。
- 小红书关键词，例如 `2027秋招 Java 后端`、`27届校招 Agent 开发`。
- 小红书账号主页或账号名。
- 用户当前已经打开的小红书可见页面。

## 输出

- 候选笔记列表：标题、作者、发布时间、URL、feed_id、xsec_token。
- 笔记详情：正文、图片、评论中的招聘线索。
- 图片 OCR 后的招聘信息。
- 公司秋招开放信号、岗位方向、届别、投递入口、可信度和待验证状态。

## 工作流程

1. 如果用户给的是关键词，调用 `xiaohongshu-mcp.search_feeds` 搜索候选笔记。
2. 如果用户给的是笔记链接或搜索结果，提取 `feed_id` 和 `xsec_token`。
3. 调用 `xiaohongshu-mcp.get_feed_detail` 获取笔记详情。
4. 如果内容在图片里，调用 `ocr.extract_text` 抽取图片文字。
5. 模型只负责从正文和 OCR 文本中抽取结构化招聘信号。
6. 程序负责去重、来源记录、可信度标记和状态流转。
7. 正式投递前必须再去企业官网、招聘官网或可信高校就业网验证。

## 权限边界

- 允许搜索、读取笔记详情和读取公开主页信息。
- 登录二维码、浏览器可见页面读取必须让用户确认。
- 禁止自动发布、评论、点赞、收藏和自动投递。
- 不绕过验证码、风控、登录限制或平台反爬机制。
- 对需要用户登录态的平台，优先使用用户可见页面和显式授权，不在后台偷偷抓取。

## 失败处理

- 如果 MCP 没有登录或缺少 `xsec_token`，返回“需要用户可见页面/登录确认”的结构化错误。
- 如果笔记详情读取失败，保留搜索结果作为候选来源，不直接生成正式岗位。
- 如果只识别出公司开放秋招但没有具体 JD，记录为招聘开放信号，后续由官网验证链路补全岗位。
