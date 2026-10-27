"""Сборка немого скринкаста из реальных кадров браузера VS Code и MLflow.

Нужны кадры и edits.json, записанные во время демонстрации через Browser Use.
Слайды отдельно находятся в docs/presentation/hw67-defense.pptx.
Скрипт сохраняет реальные изменения экрана, вырезает паузы между разделами,
добавляет главы контейнера MP4 и не создаёт звуковую дорожку.
"""
import argparse
import bisect
import json
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def ffmpeg_path(requested):
    if requested:
        return requested
    if found := shutil.which('ffmpeg'):
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise SystemExit('Укажите --ffmpeg или запустите uv run --with imageio-ffmpeg==0.6.0 python -m scripts.make_demo') from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-dir', default='submission/screencast/capture-final')
    parser.add_argument('--output', default='submission/hw67-demo.mp4')
    parser.add_argument('--ffmpeg', default=os.environ.get('FFMPEG'))
    args = parser.parse_args()
    root = Path(args.capture_dir).resolve()
    frames = [json.loads(s) for s in (root/'frames.ndjson').read_text().splitlines() if s.strip()]
    edits = json.loads((root/'edits.json').read_text())
    times = [r['time'] for r in frames]
    assert times == sorted(times), 'Нарушен порядок времени кадров'
    assert len({r['file'] for r in frames}) == len(frames), 'Повторные имена кадров'
    normalized = root/'normalized'; normalized.mkdir(exist_ok=True)
    selected, chapters, elapsed = [], [], 0.0
    for clip in edits:
        start, end = clip['start'], clip['end']
        assert end > start
        index = max(0, bisect.bisect_right(times, start)-1)
        boundaries = [start] + [t for t in times[index+1:] if start < t < end] + [end]
        for a, b in zip(boundaries, boundaries[1:]):
            if b-a < 0.001:
                continue
            i = max(0,bisect.bisect_right(times,a)-1)
            source = root/frames[i]['file']
            dest = normalized/(source.stem+'.jpg')
            if not dest.exists():
                with Image.open(source) as image:
                    image.convert('RGB').resize((1280,720),Image.Resampling.LANCZOS).save(dest,quality=95,subsampling=0)
            selected.append((dest,b-a))
        duration = end-start
        chapters.append(dict(title=clip['title'], start_seconds=round(elapsed,3), duration_seconds=round(duration,3)))
        elapsed += duration
    assert 300 <= elapsed <= 480, elapsed
    concat = root/'edited.ffconcat'
    concat.write_text('ffconcat version 1.0\n'+''.join(f"file '{p}'\nduration {d:.6f}\n" for p,d in selected)+f"file '{selected[-1][0]}'\n")
    metadata = root/'chapters.ffmetadata'
    metadata.write_text(';FFMETADATA1\n'+''.join(f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(c["start_seconds"]*1000)}\nEND={round((c["start_seconds"]+c["duration_seconds"])*1000)}\ntitle={c["title"]}\n' for c in chapters))
    output = Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    binary = ffmpeg_path(args.ffmpeg)
    subprocess.run([binary,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-i',str(metadata),
                    '-map_metadata','1','-map_chapters','1','-vf','fps=24,format=yuv420p','-an','-c:v','libx264',
                    '-preset','fast','-crf','18','-movflags','+faststart','-t',str(elapsed),str(output)],check=True)
    probe = subprocess.run([binary,'-i',str(output)],capture_output=True,text=True)
    assert 'Audio:' not in probe.stderr, 'В видео появилась звуковая дорожка'
    subprocess.run([binary,'-v','error','-i',str(output),'-f','null','-'],check=True)
    result = dict(duration_seconds=round(elapsed,3),size_bytes=output.stat().st_size,
                  width=1280,height=720,fps=24,audio_streams=0,theme='dark',
                  recorded_at='2026-10-27',capture='Actual browser screen capture of VS Code Server and MLflow',
                  editing='Reading pauses between sections trimmed; command output retained',chapters=chapters)
    Path('docs/video-metadata.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
