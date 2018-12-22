@echo off
if %time:~0,2% LSS 10 (
set filename=%date:~0,4%-%date:~5,2%-%date:~8,2%-%time:~1,2%%time:~3,2%:%time:~6,2%
) else (
set filename=%date:~0,4%-%date:~5,2%-%date:~8,2%-%time:~0,2%:%time:~3,2%:%time:~6,2%
)
@echo on
echo Time: %filename%
git remote -v
git branch
git add -A
git status
git commit -m %filename%
git push github master
git push coding master
pause