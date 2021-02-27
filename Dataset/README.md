# Data set folder
All the data should go in this folder. Please follow the given
guidelines.
1. It should be placed inside a sub-folder with the name of the model.
   e.g: If the name of the model is "Freddy", the sub folder should be
   Dataset/Freddy.

2.  The audio clips must be in .wav format. It is recommended to break
    down large audio clips to shorter segments for better results.

3. Inside the folder an excel file named "index.xlsx" must be place. The
   first column should contain the name of the audio clip (without the
   .wav extension). The second column should contain the lyrics
   corresponding to the audio clip.

    **Note:** A lyric WebScraping module was also integrated here which
    can extract lyrics through AzLyrics. It can only be done if the
    input audio contains a popular song in its entirety. Keep the second
    column blank for such songs and enter the artist and song name in
    the third and fourth columns respectively. As this relies on a third
    party lyric service and it uses long audio clips this method may not
    cause inconsistencies.

4. Similar to above, any data you want to generate or test should be
   placed in a folder named "Test" (Dataset/Test).

5. During generating and testing if custom musical notes are to be used
   for the output, enter them into another Excel file called
   "notes.xlsx". The first column should contain the starting time of
   the note in Milliseconds, and the second column must contain the note
   ending time. The third column must contain the note in the following
   format. E.g. **A4** corresponds to the A note in the 4th octave, C#5
   corresponds to the C sharp note in the 5th octave. Flat notes are not
   supported and should be instead represented using sharps.

