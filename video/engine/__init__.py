"""JewGrass Video Engine: deterministic music-video compositor.

Design: docs/video/video-engine.md. Every frame is a pure function of
(storyboard, analysis, assets, seed, t), so a render can be repeated exactly.
"""
