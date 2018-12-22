#!/bin/sh
DATE=$(date +%Y-%m-%d)
echo $DATE
git remote -v
git add -A
git status
git commit -m $DATE
git push github master
git push coding master
