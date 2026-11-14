[app]
title = ClipForge
package.name = clipforge
package.domain = org.clipforge
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json
version = 2.0
requirements = python3==3.11.6,hostpython3==3.11.6,kivy==2.3.0,numpy,yt-dlp,certifi,urllib3,charset-normalizer,idna,pyjnius,android,ffmpeg
orientation = portrait
fullscreen = 0

android.permissions = INTERNET,ACCESS_NETWORK_STATE,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE,READ_MEDIA_VIDEO
android.api = 33
android.minapi = 24
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
p4a.branch = develop

[buildozer]
log_level = 2
warn_on_root = 0
