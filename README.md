# naruto-cam

A webcam filter that replaces you with a substitution log when you do the hand seal from Naruto. Then it keeps you gone. Do the seal again and you come back.

Works in Google Meet, Zoom, Discord, Teams — anywhere you can pick a camera. It goes through OBS Virtual Camera so other apps see it as a normal webcam.

## What you need

- Windows 10 or 11
- Python 3.12 (this matters — see below)
- OBS Studio, installed but not actually running
- A webcam
- A `log.png` and a `poof.wav` (more on that in a sec)

About Python 3.12: newer MediaPipe removed the `mp.solutions` API that this uses, and there's no 3.13 wheel that still has it. So 3.13 won't work. Just grab 3.12 from python.org and check "Add Python to PATH" during install. You can keep 3.13 installed for other stuff, they don't conflict.

## Setup

Open OBS once and click **Start Virtual Camera**, then **Stop Virtual Camera**. That registers the driver with Windows. You can close OBS forever after that, it just needs to bless your machine once.

Then in a terminal:

    py -3.12 -m venv naruto-env
    naruto-env\Scripts\activate
    pip install mediapipe==0.10.21 opencv-python pyvirtualcam numpy soundfile sounddevice

## Assets

Two files go next to the script:

**log.png** — a transparent PNG of the substitution log. Search "naruto substitution log png" and grab one from pngwing or pngimg. Use their download button, not right-click save. If you end up with a fake-transparent PNG that has the gray checkerboard baked in as actual pixels, the script strips it out anyway, but a real transparent one looks better.

**poof.wav** — any anime poof sound effect. Has to be an actual WAV file. Renaming an mp3 to .wav doesn't work, and you'll spend ten minutes wondering why there's no sound while questioning every life choice that led you here.

## Run it

    python naruto_cam.py

First time, do this once: step completely out of frame (no hands, no shadow), press **b** in the preview window, then step back in. That captures your empty room. When you vanish, your silhouette gets filled with that snapshot, so it looks like you actually disappeared instead of turning into a gray blob. Skip this and you'll be a gray blob.

Then do the seal — both hands up, wrists close together, fingers overlapping. Hold it for about a quarter second. Poof. You're gone. Do it again and you're back. No log on the way back, because even ninja have limits.

In Meet / Zoom / Discord, set your camera to **OBS Virtual Camera**.

## Keys

- `b` — recapture the empty-room background
- `q` — quit

## If the seal isn't triggering

Check the yellow debug text in the preview.

- `hands: 0` → the lighting's bad. Move somewhere brighter, like Kakashi's reading spot.
- `hands: 1` → MediaPipe thinks your two hands are one hand because they're too close. Keep your fingertips about a centimeter apart, not touching.
- Two hands detected but `matching pairs` stays at 2 or 3 → lower `MIN_MATCHING_PAIRS` at the top of the script to 3.

If it triggers randomly when you're not doing anything, raise `MIN_MATCHING_PAIRS` to 5. Unless you're accidentally doing hand signs in your sleep, in which case congratulations.

## Tuning

All the constants are at the top of the file in all caps. Smoke color, smoke opacity, hold duration, transition length, all of it.

If it's laggy, set `SEG_MODEL = 0`. If it's still laggy, drop `W, H` to `640, 480`. If it's still laggy after that, it's not the code, it's your laptop.

## Why it detects wrists and knuckles instead of fingertips

First version checked fingertip positions and it never worked. When you actually clasp your hands for the seal, your index and middle fingers hide behind each other and MediaPipe loses them completely, like a jonin abandoning a genin on a mission.

So this version checks the landmarks that stay visible no matter how you cross your fingers: wrists, thumb bases, and the four knuckles. If at least 4 of the 6 pairs are close between both hands, and the wrists are close, it fires.

## Known issues

- The invisible fill is a static snapshot of your room. If you move around a lot while invisible, the edges of your silhouette won't have matching background behind them. Works best if you stay roughly in place, like a proper substitution should.
- If you bump the camera or the lighting changes, press `b` again to recapture.
- Mask edges can be a little jittery. Haven't fixed it yet.
- Python 3.13 doesn't work. Use 3.12. This is your final warning.

## Credits

- MediaPipe for the hand tracking and selfie segmentation
- OBS Studio for the virtual camera driver
- pyvirtualcam for the Python bridge
- Masashi Kishimoto for the jutsu, and for drawing hands in ways no computer vision model was ever prepared for

## License

MIT. But I would appreciate a coffee for my engineering studies cuz ik I'm not gonna finding an internship. 🥀
