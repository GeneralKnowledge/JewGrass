# Voice recording script

For training a voice model (RVC) on your own voice, so it can replace the
generated vocal on our songs. Aim for **15 to 20 minutes of clean voice in total**.
More is better, but clean beats long.

## Before you start

- **Equipment:** a phone is fine. Use a voice-memo app set to its best quality
  (WAV if offered, otherwise the highest-quality M4A). If you have a USB mic, even
  better.
- **Room:** small and soft. A bedroom with curtains, carpet and a duvet is ideal.
  Avoid kitchens and bathrooms (echo). Turn off fans, the fridge hum if you can,
  heating, TV and music. Phone on airplane mode so it doesn't buzz.
- **Distance:** phone about 20–30 cm from your mouth, slightly to the side so
  "p" and "b" sounds don't pop. Keep the same distance the whole time.
- **Level:** do a test line at your loudest (a shout from Part 6). If it distorts,
  move back a bit. Then keep that position for everything.
- **Voice:** use the voice you want the band to have. Your natural accent is
  good; don't put on a different one unless that's the character.
- **Takes:** record **one file per part** and name them `01-warmup`,
  `02-sentences`, and so on. If you fluff a line, pause, then say it again. No
  need to restart the file; the bad bits get trimmed.
- **Breaks:** drink water between parts. Stop if your voice gets tired. You can
  split this over two days.

## Part 1: Warm-up and sustained sounds (about 2 minutes)

Hold each one for about 3 seconds, at a comfortable, middle pitch:

> aah, ee, oo, oh, eh, ay, eye, ow, oy, mmm, nnn, sss, shh, zzz, vvv, rrr, lll

Then slide slowly from your lowest comfortable note to your highest and back on
"aah", twice. Then the same on "oo".

## Part 2: Sentences with lots of different sounds (about 4 minutes)

Read at a natural speaking pace, as if telling a friend:

1. The quick brown fox jumped over the lazy dog by the old stone wall.
2. Grandad's rusty van broke down by the loch on a wet Thursday in June.
3. She sells smoked fish and thick shortbread from a shed beside the pier.
4. Which witch switched the watch on the window ledge at half past nine?
5. A judge in a beige jacket measured the gorge with a plastic gauge.
6. The thirsty farmer bought three thousand feathers for thirty pence.
7. Fresh bread, warm butter and a pot of strong tea on a freezing morning.
8. The ferry's cancelled again, so the whole island queues for the bus.
9. Pour the boiling water slowly; the yellow jug is very hot.
10. Mighty heavy engines rumble through the valley after midnight.
11. Could you pass the cheese and pickle, and the big jar of mustard, please?
12. Every kid on the street knows the shopkeeper's dog by name.
13. Hush now, the baby's sleeping, and the radio's turned right down.
14. Zero, one, two, three, four, five, six, seven, eight, nine, ten.
15. Twelve, twenty, fifty-five, ninety-nine, a hundred, a thousand, a million.
16. Is it Tuesday or Wednesday? I honestly couldn't tell you.
17. Quietly, quickly, the joiner fixed the squeaky gate with a screw.
18. The vicar's voice echoed round the chapel and out over the graves.
19. Oh no! Not again! Who left the gate open?
20. Right then. Let's get this sorted, once and for all.

Then read all twenty again, **faster and more casual**.

## Part 3: The villain speaks (about 3 minutes)

Smug, posh, relaxed, enjoying yourself. These are the spoken intros and
outros from our songs:

> Raise a glass, boys. To the coal that ain't mine no more, the creek I sold twice, and every family that signed without reading. Ladies and gentlemen... your landlord.

> Charge your glasses. To my lads in the City, my cousin at the Treasury, my godson at the paper, the Viscount, the hedge fund, and the wee island where the money sleeps. And to you, the tenant.

> Raise a dram, gentlemen. To the glen my great-great-grandfather emptied, the river I rent out by the fish, and every village that voted to let me keep it.

> Good evening, shareholders. Another record year. Bills up, profits up, morale... irrelevant.

> Welcome, shareholders, to the AGM. Record dividends. Record bonuses. Record spills... let's not dwell.

> Y'all have a blessed day now. Rent's due Friday.

> Mind the damp, now. Rent's due Friday.

> Do enjoy the scenery. Rent's due Friday.

> Stay warm out there. Direct debit's due Friday.

Read each one twice: once quiet and creepy, once loud and showy.

## Part 4: Rap verses (about 3 minutes)

Rap these over a steady beat. Tap your foot or play a metronome at 140 BPM in
headphones (**not** out loud, or it gets into the recording). Don't worry about
being perfect; we want your rhythm and attitude.

> I own the flat that you rent and the bank with your loan
> The water in your tap and the signal on your phone
> Sewage in the river? That's a dividend, son
> Fined a million quid and I made a hundred and one
> Black mould on the ceiling, black mould on the cot
> Letting agent says ventilate, and that's all that you got

> Great-great-granddad cleared the glen for sheep
> Burned the roofs so the tenants couldnae sleep
> Put 'em on a boat with a Bible and a cough
> Now I charge the tourists twelve quid to see where they set off

> I pumped the oil from the sea off the coast
> Shipped the money south and I raised a wee toast
> Fifty years of black gold, so what have you got?
> A pothole, a food bank and a boarded-up shop

> They sold me the pipes back in eighty-nine
> Every reservoir, every drain down the line
> Borrowed ten billion, paid it out to the board
> Now the debt's on your bill and my yacht's in a fjord

Do each verse twice: once laid back, once aggressive.

## Part 5: Singing (about 3 minutes)

Sing these slowly, any simple tune you like. Even if you don't think of yourself
as a singer, this matters: the model needs to hear your voice holding notes.

> Bless the laird who owns the glen
> Bless his hand and his fountain pen
> We bow our heads and we pay the rent
> And the Lord alone knows where the money went

> Oh, the creek don't remember
> Who went under, who came home
> It just runs through the holler
> Singing low, singing low

> We bought the hill! We bought the shore!
> We bought the burn and the bothy door!
> Every penny from a jam jar spill
> Raise your glass, we bought the hill!

Sing each once quietly and once with full voice. Then sing any song you know
well, for a minute or so, with no backing track.

## Part 6: Chants and shouts (about 1 minute)

Short, loud, like a crowd at a gig. Step back a little first.

> I own the county! Own the creek!
> Own the glen! Own the loch!
> Pay the bill! Cold hame!
> Landlord! Rent's due!
> Swim at your own risk!

## Part 7: Free talk (about 2 minutes)

Just talk naturally about anything: your day, a film, what you want the band to
sound like. This gives the model your everyday voice.

## Sending the files to the laptop

Copy the files to your computer first, then:

```
ssh red@create 'mkdir -p ~/band/voice/raw'
scp 01-warmup.* 02-sentences.* 03-villain.* 04-rap.* 05-singing.* 06-chants.* 07-talk.* red@create:~/band/voice/raw/
```

If `create` doesn't resolve, use the laptop's IP address. Then tell me they're
there, and I'll clean them up, train the voice model and try it on a song.

## A note on use

This is for your own voice. Only train on recordings of yourself, or of someone
who has agreed in writing to be in the band's sound.
