from pathlib import Path
from unittest import TestCase


class FrontendNAsciiBackgroundTest(TestCase):
    def test_n_ascii_component_is_customizable_and_shadcn_ready(self) -> None:
        component_path = Path("apps/web/src/components/ui/n-ascii.tsx")
        self.assertTrue(component_path.exists())

        component_source = component_path.read_text(encoding="utf-8")

        self.assertIn("export interface AsciiArtProps", component_source)
        self.assertIn("rows?: number", component_source)
        self.assertIn("density?:", component_source)
        self.assertIn("intensity?: number", component_source)
        self.assertIn("className?: string", component_source)
        self.assertIn("style?: CSSProperties", component_source)
        self.assertIn("lucide-react", component_source)
        self.assertIn("Cpu", component_source)
        self.assertIn("Sparkles", component_source)
        self.assertIn("border-border", component_source)
        self.assertIn("bg-background", component_source)
        self.assertIn("text-muted-foreground", component_source)
        self.assertIn("buildAsciiRows", component_source)
        self.assertNotIn("dangerouslySetInnerHTML", component_source)

    def test_app_wires_n_ascii_as_background_layer(self) -> None:
        app_source = Path("apps/web/src/app/App.tsx").read_text(encoding="utf-8")
        style_source = Path("apps/web/src/styles/global.css").read_text(encoding="utf-8")

        self.assertIn('import { AsciiArt } from "../components/ui/n-ascii"', app_source)
        self.assertIn("app-ascii-background", app_source)
        self.assertIn("<AsciiArt", app_source)
        self.assertIn('aria-hidden="true"', app_source)
        self.assertIn("showHeader={false}", app_source)
        self.assertIn(".app-ascii-background", style_source)
        self.assertIn(".n-ascii", style_source)
        self.assertIn("--background", style_source)
        self.assertIn("--foreground", style_source)
        self.assertIn("--border", style_source)
        self.assertIn("@keyframes n-ascii-shimmer", style_source)
