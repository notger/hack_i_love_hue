# Overview

Small repository to explore automated options to solve a colour-based puzzle as seen in the mobile phone game "I love Hue".
In this game, you have to sort colours with as little steps as possible. My daughter currently loves it (probably not anymore by the time you are reading this) and it got me curious: Which ways are there to solve this and is there a consistently reachable lower boundary of minimum steps?

# Installation

Run `pip install -r requirements.txt` to get all necessary libraries and remember: Python 3.x, of course. ;)

# Usage

I opted for a flat structure, so there is one main-file that runs the show. Everything should be triggered from the root directory, to save you the hassle of defining paths or installing this as a module and me the hassle of adding stupid path-hacks to the files.
Just run `python3 -m main` and you should be good. If you want to only analyse the lastest image, then add the flag `--latest`.

All images you want to analyse go into the `images` sub-folder and should be of the jpeg-format. They should be made as screenshots from the phone you are playing on (in case you actually want to use this to solve a puzzle).
You will find a slideshow of the step-by-step-solutions in the `solutions`-folder, with a filename corresponding to the input filename. Just open the gif with a gifviewer which allows you to manually control the frames and you should be good.
Be sure to follow the instructions minutely. If you misclick, you are done for.

Please remember that this is just a little fun project I want to build around the image-part and the solution part. This means the delivering the solution to me is totally an afterthought and not the point. There are plenty of ways this could be more accessible, but given that this is for my use only, I am fine with this way.

Please also don't add images to the `images`-folder and then push them. The images-folder is only here so that you have a default-image to work on when you pull this repo.

# Tests

Run `pytest` from the root directory. Tests will fail if invoked otherwise, as I was lazy with the paths.
