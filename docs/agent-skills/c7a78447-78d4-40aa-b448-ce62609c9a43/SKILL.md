---
name: database-operations
description: 当用户询问 OfferMaster 本地企业、岗位、岗位来源或招聘信号时使用；查询可以自动执行，修改公司资料和标记岗位线索无效必须先让用户确认。
source_types:
  - agent_chat
  - job_discovery
required_tools:
  - database.company_search
  - database.company_profile
  - database.job_search
  - database.source_search
allowed_tools:
  - database.company_search
  - database.company_profile
  - database.job_search
  - database.source_search
ask_tools:
  - database.company_update
  - database.job_lead_delete
disallowed_tools:
  - raw_sql.execute
  - database.direct_delete
compatibility:
  - offermaster
license: Internal
---

# 本地数据库操作

这个 Skill 负责把用户关于本地企业库、岗位库和招聘来源库的问题交给固定的数据库工具处理。

## 适用场景

- 用户问数据库里有没有某家公司，或者想看某家公司的完整档案。
- 用户按公司、岗位、城市、岗位类型或关键词查询本地正式岗位。
- 用户想查看岗位来源、来源类型、可信度和来源下的线索数量。
- 用户要求修改公司资料或删除岗位线索时，先展示即将修改的对象和字段，再请求确认。

## 工作流程

1. 先选择最具体的查询工具，不让模型拼接 SQL。
2. 公司查询要同时核对正式企业表、正式岗位表、岗位线索表和招聘信号表。
3. 返回结果时说明数据来自哪张业务表，不能把招聘信号直接说成正式岗位。
4. 公司更新只允许修改白名单字段，并且必须经过用户确认。
5. 岗位线索删除默认采用软删除，保留原记录和审计说明，并且必须经过用户确认。

## 权限边界

- 允许自动执行四个只读查询工具。
- `database.company_update` 和 `database.job_lead_delete` 只能在用户确认后执行。
- 禁止执行原始 SQL、绕过工具注册表直接写数据库或物理删除审计记录。
- 查询结果为空时如实说明，不使用外部网页信息补造本地数据库内容。

## 输出要求

- 公司查询返回是否存在、各业务表的数量和可追溯记录标识。
- 公司详情分别展示正式企业、正式岗位、岗位线索和招聘信号。
- 岗位和来源查询返回筛选条件、数量和来源信息。
- 修改或删除返回实际影响对象、变更字段、操作模式和结果状态。
