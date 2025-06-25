@echo off
setlocal EnableDelayedExpansion

:: Set Desktop Path
set "DESKTOP=%USERPROFILE%\Desktop"
cd /d "%DESKTOP%"

:: Create or clear log
set "LOG=%DESKTOP%\organize_log.txt"
echo [%DATE% %TIME%] Organizing Desktop files... > "%LOG%"

:: Define file categories
set "Documents=pdf doc docx xls xlsx ppt pptx txt rtf odt ods odp csv md markdown tex log chm lit wps wpd"
set "Images=jpg jpeg png gif bmp tiff tif svg webp heic ico avif raw dng cr2 nef orf arw raf sr2 pef"
set "Videos=mp4 avi mov mkv wmv flv m4v webm vob mpg mpeg 3gp ts rm m2ts mts"
set "Audio=mp3 wav aac flac ogg m4a wma opus aiff mid midi amr au"
set "Archives=zip rar 7z tar gz bz2 xz iso cab arj lzh ace sit tgz zpaq xar"
set "Executables=exe msi bat cmd com jar elf"
set "Scripts=py js jsx ts tsx php html htm css scss sh ps1 psd vbs lua rb cs c cpp h java go rs"
set "3DModels=obj fbx blend stl 3ds dae glb gltf ply ma mb abc usd usdz"
set "CAD=dxf dwg step stp iges igs sldprt sldasm sat"
set "Data=json xml yaml yml ini cfg dat db sqlite sql sav rdata feather parquet h5 hdf5 npy pkl"
set "Other=other"

:: Create folders
for %%C in (Documents Images Videos Audio Archives Executables Scripts 3DModels CAD Data Other) do (
    if not exist "%%C" mkdir "%%C"
)

:: Start organizing files
for %%F in (%DESKTOP%\*.*) do (
    if not exist "%%F\" (
        set "ext=%%~xF"
        set "ext=!ext:~1!"
        set "category=Other"
        for %%C in (Documents Images Videos Audio Archives Executables Scripts 3DModels CAD Data) do (
            for %%E in (!%%C!) do (
                if /I "!ext!"=="%%E" (
                    set "category=%%C"
                )
            )
        )
        move "%%~nxF" "!category!\" >nul 2>&1
        echo Moved %%~nxF to !category!\ >> "%LOG%"
    )
)

echo Done! All files have been organized. Log saved to organize_log.txt.
pause
