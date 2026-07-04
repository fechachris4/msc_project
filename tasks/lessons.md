# Lessons

- Pattern: A scene can fail at a MuJoCo XML line even when the XML syntax is valid if that line references an external model path that does not exist locally.
  Rule: For MJCF asset/model load errors, first verify the referenced file path exists, then run MuJoCo `compile` against the parent scene to prove nested assets load.

- Pattern: Auto-generating opaque numeric blobs (e.g. a MuJoCo keyframe with 28 pasted qpos/ctrl values) into a hand-maintained XML file got rejected as unreadable, even though it was functionally correct.
  Rule: Keep magic-number vectors out of hand-edited MJCF. When a value can only be expressed numerically in XML, prefer composing it at load time in Python from named sources (e.g. merging inherited keyframes by name); if a literal must stay in XML, it needs a comment deriving where each number came from.

- Pattern: Replacing `tasks/todo.md` wholesale during a small planning update risks clobbering prior review history.
  Rule: For task tracking files, preserve existing sections and use targeted patches that append or update only the active checklist/review.
