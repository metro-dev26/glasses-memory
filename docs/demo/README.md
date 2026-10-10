# Demo video: sample5

66 s, first-person, phone held like glasses, 1080p 120 fps. The footage is
personal, so it is not in the repo: get the original file from Abhinav (not a
re-compressed copy, or the numbers below won't match) and save it as
`videos/sample5.mp4`. Then:

```bash
python -m engine.run videos/sample5.mp4 --store store/sample5 --fresh
```

## What moves (checked by the recorder)

| Object | From | To | When |
|---|---|---|---|
| Stapler | desk | bed | ~52 s |
| "Best Love Poems" book | desk | floor, by the mat | ~46 s |
| Red tripod | side table | floor | ~58-60 s |
| Orange perfume bottle | (not on the desk at 14 s) | desk, by the coins | ~38 s |
| Prime box | desk corner | same place | did not move |
| Coin | desk | picked up, put back | did not move |

## First run (engine at commit 179496b)

654 frames at 14.5 fps, 83 objects in memory, 90 re-identifications.
The three objects that changed place were each seen in their new place, but
came back as new objects instead of re-identified ones: the stapler on the bed
as #74 "iphone", the book on the floor as #69 "ticket", the tripod on the floor
as #79 / #80. A moved-object check on top of this would report "missing" plus
"new", never "moved".
