# AFTER HOURS

A four-member contemporary boy band (Reuben's spec, Nov 14): classic boy band songwriting instincts with
contemporary production. Their selling point is the blend: each voice is recognisable, but the group is
better together. Everything is original; genre descriptors only, no real artist named or imitated.
Separate from LOWBRIDGE and from the earlier Matching Jackets sketch (kept in `matching-jackets/`).

| Member | Role | Voice |
|---|---|---|
| **Alex** | the lead | expressive mid-to-high tenor, slight rasp at the top |
| **Jamie** | the high voice | bright high tenor, clean falsetto, answers and ad-libs |
| **Noah** | the foundation | warm intimate baritone, a real lower line |
| **Leo** | the colour | smooth soulful tenor, gentle runs, pre-chorus lifts |

## Songs

| # | Song | Tests | Tempo / key | Length | Status |
|---|---|---|---|---|---|
| 1 | [One More Night](design/one-more-night.yaml) | catchy pop, does it sound like a group | 112, A major | 3:45 | round 1 |
| 2 | [If You Were Here](design/if-you-were-here.yaml) | harmony, a cappella bridge, emotion | 76, D major | 4:00 | round 1 |
| 3 | [Everybody Knows](design/everybody-knows.yaml) | member interplay, dance-pop | 120, E major | 3:35 | round 3 |

Following the project rules, every song starts on a voice (no instrumental intro) and ends sung. Each
design's `changes_from_spec` lists what moved from the original spec and why.

```
python3 after-hours/build.py     # rebuild tracks.yaml after editing a design; exits 1 on any warning
```

## Round 1 (Kaggle, XL turbo)

```
TRACKS_YAML = "https://raw.githubusercontent.com/GeneralKnowledge/JewGrass/claude/project-thread-hhd444/after-hours/tracks.yaml"
TRACKS = ["ah-one-more-night", "ah-if-you-were-here"]
SEEDS = {"ah-one-more-night": [9601, 9602], "ah-if-you-were-here": [9601, 9602]}
```

The main test: does the chorus sound like a group, or like one voice copied four times? ACE-Step gets
member identities only as words in the captions and section tags (the `performer` field is metadata for a
future vocal-aware compiler). If the voices blend into one, round 2 renders sections separately per member and
assembles them on the laptop, as planned for LOWBRIDGE.
