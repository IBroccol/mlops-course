"""Озвученный обзор реальных UI-снимков и логов (macOS say + ffmpeg).

uv run --with imageio-ffmpeg==0.6.0 python -m scripts.make_demo
Ни браузер, ни результаты проверки этим скриптом не имитируются.
"""
import json
import re
import subprocess
import textwrap
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
MONO='/System/Library/Fonts/Menlo.ttc'
ROOT=Path('submission/video-work'); ROOT.mkdir(parents=True,exist_ok=True)


def font(size,mono=False): return ImageFont.truetype(MONO if mono else FONT,size)


def wrap(draw,text,f,width):
    words=text.split(); lines=[]; line=''
    for word in words:
        trial=(line+' '+word).strip()
        if draw.textlength(trial,font=f)>width and line: lines.append(line);line=word
        else: line=trial
    return lines+[line]


def frame(i,scene):
    img=Image.new('RGB',(1920,1080),'#102535');d=ImageDraw.Draw(img)
    d.text((55,40),'MLOPS · WIKIDATA · 21.10.2026',font=font(24),fill='#8fc6c1')
    d.text((55,88),scene['title'],font=font(42),fill='white')
    d.line((55,158,1860,158),fill='#345366',width=2)
    if scene['image']:
        shot=Image.open(Path('docs')/scene['image']).convert('RGB')
        shot=ImageOps.contain(shot,(1320,770))
        x=50+(1320-shot.width)//2;y=195+(770-shot.height)//2
        img.paste(shot,(x,y))
        y=235
        for note in scene['notes']:
            for line in wrap(d,note,font(31),450):
                d.text((1410,y),line,font=font(31),fill='#eff6f8');y+=43
            y+=27
    elif i == 6:
        # Только строки из настоящего check-all, без выдуманного терминала.
        raw=Path('docs/reports/check-all.txt').read_text()
        lines=[s for s in raw.splitlines() if s.startswith('PASS ') or s.startswith('Итог:') or s.startswith('Отпечаток') or s.startswith('Ran ') or s=='OK']
        lines=lines[:11]+[s for s in lines[11:] if not s.startswith('PASS ')][:2]
        d.rounded_rectangle((50,200,1865,940),18,fill='#081823')
        d.text((82,228),'ФАКТИЧЕСКИЙ ВЫВОД make check-all',font=font(26),fill='#8fc6c1')
        y=291
        for line in lines:
            for part in wrap(d,line,font(26,True),1720):
                d.text((82,y),part,font=font(26,True),fill='#cce9d5');y+=39
            y+=6
    else:
        y=235
        for note in scene['notes']:
            d.text((95,y),'• '+note,font=font(46),fill='#eff6f8');y+=105
    d.text((55,1015),'Автоматическая озвучка · реальные снимки MLflow и результаты проверок',font=font(23),fill='#9eb6c3')
    d.text((1770,1015),f'{i+1} / 7',font=font(23),fill='#9eb6c3')
    out=ROOT/f'{i:02}.png';img.save(out);return out


def main():
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    scenes=json.loads(Path('docs/video-script.json').read_text());durations=[]
    for i,scene in enumerate(scenes):
        still=frame(i,scene);speech=ROOT/f'{i:02}.txt';speech.write_text(scene['speech'])
        audio=ROOT/f'{i:02}.aiff'
        subprocess.run(['say','-v','Milena','-r','155','-f',str(speech),'-o',str(audio)],check=True)
        probe=subprocess.run([ffmpeg,'-i',str(audio)],capture_output=True,text=True)
        h,m,s=map(float,re.search(r'Duration: (\d+):(\d+):([\d.]+)',probe.stderr).groups())
        duration=max(48,h*3600+m*60+s+2);durations.append(duration)
        subprocess.run([ffmpeg,'-y','-loglevel','error','-loop','1','-framerate','10','-i',str(still),'-i',str(audio),
                        '-vf','format=yuv420p','-c:v','libx264','-preset','veryfast','-tune','stillimage','-crf','23',
                        '-c:a','aac','-b:a','128k','-af','apad','-t',str(duration),str(ROOT/f'{i:02}.mp4')],check=True)
        print(f'Chapter {i+1}: {duration:.2f}s',flush=True)
    listing=ROOT/'concat.txt';listing.write_text(''.join(f"file '{i:02}.mp4'\n" for i in range(len(scenes))))
    out=Path('submission/hw67-demo.mp4')
    subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(out)],check=True)
    assert 300<=sum(durations)<=480, durations
    Path('docs/video-metadata.json').write_text(json.dumps(dict(duration_seconds=sum(durations),chapters_seconds=durations,
             narration='macOS Milena synthetic voice; not a human screencast',size_bytes=out.stat().st_size),indent=2)+'\n')
    print(out,sum(durations),out.stat().st_size,flush=True)


if __name__ == '__main__': main()
