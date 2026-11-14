# Voice recording script (short version, about 5 minutes)

For training a voice model (RVC) so that **your voice becomes the band's
singing voice**. The plan: we keep generating the songs as now, then the model
swaps the generated singer's voice for yours.

## You don't need to be able to sing

The model copies the **sound** of your voice (its tone and accent). The tune,
timing and pitch all come from the generated vocal it's replacing. So the songs
won't be out of tune just because you are.

What it does need is to have heard your voice **holding notes, high and low**.
If it has only ever heard you talking, sung parts come out thin, wobbly or
robotic, especially high notes. So in this script you'll hold notes and sing a
few lines. Sing badly; that's fine. Wobbly and off-key is useful. Silent is not.

**About length:** 5 minutes is the minimum that works. Expect a decent result on
rap and low or mid singing, and some roughness on high or long held notes. If
we want it better after the first try, record this script a second time on
another day (10 minutes total) and I retrain. That's the biggest single
improvement.

## Before you start

- **Phone is fine.** Voice memo app, best quality setting. Hold it 20–30 cm
  from your mouth, slightly to the side. Same distance throughout.
- **Quiet, soft room.** Bedroom with curtains and a duvet is ideal. No kitchen or
  bathroom (echo). Fans, TV and music off. Phone on airplane mode.
- **No music playing out loud.** If you want a beat or a tune to follow, use
  headphones, so it doesn't get into the recording.
- **Test the loudest part first** (Part 6). If it crackles, move the phone back
  a bit and keep it there.
- **One recording per part**, named with the part number (`1-warmup`,
  `2-notes`, ...). If you mess up, pause and carry on. Don't restart.
- Warm up properly, sip water, and stop if your throat gets sore.

## Part 1: Warm-up slides (about 45 seconds)

Slide on "aah" from your lowest comfortable note up to your highest and back
down, slowly. Do it three times. Then the same on "oo", then on "ee".
Going a bit croaky at the top is fine.

## Part 2: Held notes (about 1 minute)

Pick a low note, a middle note and a high-ish note (any notes, they don't have
to match anything). On each of those three notes, hold each vowel for about
3 seconds:

> aah, ee, oo, oh, ay

Then hum "mmm" on each note for 3 seconds.

## Part 3: Spoken sentences (about 45 seconds)

Natural pace, as if telling a pal. These cover the rest of the sounds in
English and your accent:

1. Grandad's rusty van broke down by the loch on a wet Thursday in June.
2. Which witch switched the watch on the window ledge at half past nine?
3. A judge in a beige jacket measured the gorge with a plastic gauge.
4. The thirsty farmer bought three thousand feathers for thirty pence.
5. It's a braw bright night, but the wind off the hill would cut ye in two.
6. Could you pass the cheese and pickle, and the big jar of mustard, please?
7. Hush now, the wean's sleeping, and the radio's turned right down.
8. Is it Tuesday or Wednesday? I honestly couldnae tell you.

## Part 4: Singing (about 1 minute 30)

Sing these slowly to any simple tune you like (make one up). Stretch the long
words out. Sing each verse **twice**: first quietly, then with full voice.

> Bless the laird who owns the glen
> Bless his hand and his fountain pen
> We bow our heads and we pay the rent
> And the Lord alone knows where the money went

> We bought the hill! We bought the shore!
> We bought the burn and the bothy door!
> Every penny from a jam jar spill
> Raise your glass, we bought the hill!

Then sing a song you know well, for about 30 seconds, with no backing track.

## Part 5: Rap (about 45 seconds)

Tap your foot or use a metronome at 140 BPM **in headphones**. Say it in time
with attitude; it doesn't need to be good. Once laid back, once aggressive.

> I own the flat that you rent and the bank with your loan
> The water in your tap and the signal on your phone
> Sewage in the river? That's a dividend, son
> Fined a million quid and I made a hundred and one

> Great-great-granddad cleared the glen for sheep
> Burned the roofs so the tenants couldnae sleep
> Put 'em on a boat with a Bible and a cough
> Now I charge the tourists twelve quid to see where they set off

## Part 6: Chants and shouts (about 20 seconds)

Like a crowd at a gig. Step back a little first. Hold the last word of each
line as a long shout.

> Own the glen! Own the loooch!
> Pay the bill! Cold haaame!
> Landlord! Rent's duuue!
> Swim at your own riiisk!

## Sending the files to the laptop

Copy them to your computer, then:

```
ssh red@create 'mkdir -p ~/band/voice/raw'
scp 1-* 2-* 3-* 4-* 5-* 6-* red@create:~/band/voice/raw/
```

If `create` doesn't resolve, use the laptop's IP address. Then tell me they're
there, and I'll clean them up, train the voice model and try it on a song.

## A note on use

This is for your own voice. Only train on recordings of yourself, or of someone
who has agreed in writing to be in the band's sound.
