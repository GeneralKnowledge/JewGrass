# Voice recording script (short version, about 5 minutes)

For training a voice model (RVC) on your own voice. About **5 minutes of clean
reading** in total. That's on the low side for RVC but workable: it gives a
usable model for speech and rap-style lines. If we later want it to hold sung
notes better, you can add another 5 minutes and we retrain.

No singing or rapping skill needed. Just read it in your own voice and accent.

## Before you start (quick)

- **Phone is fine.** Voice memo app, best quality setting. About a hand's width
  and a half (20–30 cm) from your mouth, slightly to the side. Same distance
  throughout.
- **Quiet, soft room.** Bedroom with curtains and a duvet is ideal. No kitchen or
  bathroom (echo). Fans, TV and music off. Phone on airplane mode.
- **Test the loud line first** (the shouts in Part 6). If it crackles, move the
  phone back a bit and keep it there.
- **One take per part.** Make a new recording for each part and name it with the
  part number (`1-warmup`, `2-sentences`, ...). If you fluff a line, pause and
  say it again. Don't restart; the bad bits get trimmed.
- Sip water between parts. Leave a second of silence at the start of each file.

## Part 1: Warm-up (about 30 seconds)

Hold each sound for about 2 seconds:

> aah, ee, oo, oh, ay, mmm, sss, shh, rrr, lll

Then slide on "aah" from your lowest comfortable note up to your highest and
back down, once. It doesn't need to sound nice.

## Part 2: Sentences (about 1 minute 30)

Natural pace, as if telling a pal:

1. Grandad's rusty van broke down by the loch on a wet Thursday in June.
2. She sells smoked fish and thick shortbread from a shed beside the pier.
3. Which witch switched the watch on the window ledge at half past nine?
4. A judge in a beige jacket measured the gorge with a plastic gauge.
5. The thirsty farmer bought three thousand feathers for thirty pence.
6. The ferry's cancelled again, so the whole island queues for the bus.
7. It's a braw bright night, but the wind off the hill would cut ye in two.
8. Pour the boiling water slowly; the yellow jug is very hot.
9. Could you pass the cheese and pickle, and the big jar of mustard, please?
10. Hush now, the wean's sleeping, and the radio's turned right down.
11. Twelve, twenty, fifty-five, ninety-nine, a hundred, a thousand, a million.
12. Is it Tuesday or Wednesday? I honestly couldnae tell you.

## Part 3: The villain (about 45 seconds)

Read each line **twice**: first quiet and creepy, almost a whisper, then loud
and showy like you're at a party.

> Raise a dram, gentlemen. To the glen my great-great-grandfather emptied, and every village that voted to let me keep it.

> Good evening, shareholders. Another record year. Bills up, profits up, morale... irrelevant.

> Do enjoy the scenery. Rent's due Friday.

## Part 4: In rhythm (about 30 seconds)

Say this to a steady beat (tap your foot). Not rapping, just speaking in time,
with attitude. Once laid back, once angry.

> Great-great-granddad cleared the glen for sheep
> Burned the roofs so the tenants couldnae sleep
> Put 'em on a boat with a Bible and a cough
> Now I charge the tourists twelve quid to see where they set off

## Part 5: Ewan's voicemails (about 1 minute 30)

These go on the album as they are (tracks 4 and 6), so they do double duty.
Ewan is a young crofter leaving messages for his sister Morag. Plain, natural,
a bit fed up. Read them like a real voicemail: phone a bit closer than before.
Do each one twice and we'll keep the better take. Don't say the bits in
brackets.

**Track 4: "They've Sold It"** (spring, half joking)

> Morag, it's me. You'll not believe this.
> He's sold it. The whole glen.
> The laird. Sold the lot.
> Nobody knows who to.
> Mrs Grant at the shop says it's a bank.
> Davie says it's a film star.
> Somebody said "a fund", whatever that is.
> Anyway. Spring's here, the lambs are out,
> and apparently we belong to somebody new.
> Phone me back.

**Track 6: "Four Hundred Companies"** (summer, starting to dig)

> Morag, it's me again. Right. I did what you said,
> I looked them up on that Companies House thing.
> The buyer's called Corran Natural Capital L P.
> A limited partnership. Scottish, would you believe.
> Registered to a wee office in Glasgow. Above a vape shop.
> And Morag... there's four hundred companies at that address.
> Four hundred. In one room. Above a vape shop.
> (pause)
> So who owns it? Another company.
> Who owns that? Something in Belize.
> I'm sat in the van wi' the midges, reading about Belize.
> Oh, and they've put a gate on the hill path.
> Phone me back. I mean it this time.

## Part 6: Shouts and whispers (about 20 seconds)

Step back a little for the shouts.

> (shout) Own the glen! Own the loch!
> (shout) Landlord! Rent's due!
> (shout) Swim at your own risk!
> (whisper) Nobody here owns anything. That's rather the point.
> (whisper) Phone me back.

## Part 7: Just talk (about 30 seconds)

Talk about anything: your day, what you had for breakfast, what you want the
band to sound like. This gives the model your everyday voice.

## Sending the files to the laptop

Copy them to your computer, then:

```
ssh red@create 'mkdir -p ~/band/voice/raw'
scp 1-* 2-* 3-* 4-* 5-* 6-* 7-* red@create:~/band/voice/raw/
```

If `create` doesn't resolve, use the laptop's IP address. Then tell me they're
there, and I'll clean them up, train the voice model and try it on a song. The
Ewan takes from Part 5 also get cleaned up and saved for tracks 4 and 6.

## A note on use

This is for your own voice. Only train on recordings of yourself, or of someone
who has agreed in writing to be in the band's sound.
