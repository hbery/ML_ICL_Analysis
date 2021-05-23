
# Preparation of input data

## Downloading datasets
Downscaled sample datasets (512x512px) are available under [TODO]

### NASA dataset
https://photojournal.jpl.nasa.gov

1. Select sub-dataset (e.g Universe)
2. `curl https://photojournal.jpl.nasa.gov/gallery/universe?start=400 | tee tmp.txt | grep -Eo 'PIA(.*).tif' | grep -iv .jpg >> input_universe.txt`
3. Create direct link by adding `https://photojournal.jpl.nasa.gov/tiff/` on the beginning of each line.
4. Download all images `xargs -n 1 curl -O --max-filesize 52428800 < input_universe.txt` (50MB to avoid downloading huge images)
5. Manually delete synthetic or rubbish images

### RAISE dataset
Nature dataset was used
http://loki.disi.unitn.it/RAISE/download.html#

1. Request the dataset and download csv file
2. Extract direct download links `cat RAISE_264.csv | grep -Eo 'http:\/\/193.205.194.113\/RAISE\/TIFF\/(.*).TIF' > input_RAISE.txt`
3. Download all images `wget -i input_RAISE.txt`


## Image rescaling

### Downscaling to 224x224px

#### Downscaling to single-dimension 224px (to avoid distorions)
1. Download IfranView https://www.irfanview.com
2. Open batch conversion/rename and load input images
3. Select output format to *bmp*. If you are using newer encoder (e.g. libjpeg-turbo) then you can use *png* (do not forget to change filters in scripts below)
4. Tick checkbox ***Use advanced options (for bulk resize)*** and click ***Advanced*** button
5. In new window check only ***RESIZE*** checkbox group
6. ***Set one or both sides to 224x224px***
7. Tick these 3 checkboxes: *This is the minimal size (if both sizes set)*, *Preserve aspect ratio (proportional)*, *Use resample function (better quality)* and click *OK*
8. Set output folder and click *Start batch*

#### Cropping to 224x224px

Now we have to crop images to 224x224px. We cannot crop all images in the same way.
We will crop images using different starting points (50% from center and 12,5% per each corner)

1. Open Powershell in directory above and run this script (Powershell is cross-platform :))

```PS
$i=1
gci [images_dirname] | Select FullName | ForEach-Object {
        if ($i % 8 -eq 0) {
            Add-Content -Path .\center.txt -Value $_.FullName }
        elseif ($i % 8 -eq 1) {
            Add-Content -Path .\up_left.txt -Value $_.FullName }
        elseif ($i % 8 -eq 2) {
            Add-Content -Path .\center.txt -Value $_.FullName }
        elseif ($i % 8 -eq 3) {
            Add-Content -Path .\up_right.txt -Value $_.FullName }
        elseif ($i % 8 -eq 4) {
            Add-Content -Path .\center.txt -Value $_.FullName }
        elseif ($i % 8 -eq 5) {
            Add-Content -Path .\down_left.txt -Value $_.FullName }
        elseif ($i % 8 -eq 6) {
            Add-Content -Path .\center.txt -Value $_.FullName }
        elseif ($i % 8 -eq 7) {
            Add-Content -Path .\down_right.txt -Value $_.FullName }    
        $i = $i + 1 
}
```
2. Open batch processing in IfranView
3. In advances setting uncheck the *RESIZE* checkbox and select only *CROP* checkbox. Select width and height to 224px.
4. Load txt respectively to the start corner selected in crop settings.
5. Save all images from this step to single directory.


**Archive all images (they might be used multiple times in the future)**

## Image conversion

1. Download encoder - https://github.com/LuaDist/libjpeg
2. Add `bin` directory to the Path (cjpeg.exe will be used)
3. Open Powershell window in input images directory
4. Run script below to encode and categorize images

```PS
$qualities = 5,15,30,50,80,100
foreach ($q in $qualities)
{
    $qname = [string]('{0:d2}' -f [int]$q)
    New-Item -ItemType Directory -Force -Path $qname
}
gci -Filter "*.bmp"| foreach {
    foreach ($q in $qualities)
    {
        $qname = [string]('{0:d2}' -f [int]$q)
        cjpeg.exe -quality $q -block 8 -outfile (Join-Path -Path "$qname" -ChildPath "$((Get-Item $_.Name).BaseName)_$qname.jpg") $_.Name
        Write-Host "$((Get-Item $_.Name).BaseName)_$qname.jpg"
    }
}
```


