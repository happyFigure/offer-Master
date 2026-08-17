import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class XiaohongshuVisiblePageTest(unittest.TestCase):
    def test_extracts_note_body_and_defers_image_parsing_from_visible_text(self):
        from app.domains.jobs.xiaohongshu_visible_page import parse_xiaohongshu_visible_text

        visible_text = """
登录后推荐更懂你的笔记
小红书
手机号登录
1/3
悲伤土豆鸡肉饭
关注
27 届秋招 | 七月总结
最近陆续有一些公司开了秋招，但整体来说开的数量还不算特别多。
面试/流程汇总
小红书暑期实习   7.8一面 7.13 二面 已 OC 拒
腾讯 wxg实习       7.24 一面 7.27 二面（挂）
米哈游 秋招          7.22 一面  7.28 二面（已挂）
07-31 四川
共 37 条评论
登录查看全部评论内容
"""

        parsed = parse_xiaohongshu_visible_text(
            visible_text,
            page_title="27 届秋招 | 七月总结 - 小红书",
        )

        self.assertEqual("27 届秋招 | 七月总结", parsed.title)
        self.assertIn("最近陆续有一些公司开了秋招", parsed.text)
        self.assertIn("米哈游 秋招", parsed.text)
        self.assertNotIn("登录后推荐", parsed.text)
        self.assertNotIn("共 37 条评论", parsed.text)
        self.assertEqual(3, parsed.image_count)
        self.assertTrue(parsed.image_parse_deferred)


if __name__ == "__main__":
    unittest.main()
