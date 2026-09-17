Add-Type -AssemblyName System.Speech
# line, SAPI rate (part 2 is faster)
$lines = @(
 @("In sixteen sixty-two, Robert Hooke was appointed Curator of Experiments.", 1),
 @("He was the first person in human history to make a living by experimenting.", 1),
 @("", 0),
 @("Three years later, he discovered the cell.", 1),
 @("Now, in twenty twenty-six, a new golden age of discovery begins.", 2),
 @("We are building his successors. Not just one, but countless scientists.", 2),
 @("That never clock out. Around the clock. Seven days a week.", 2),
 @("Robot Scientists, accelerating scientific discovery,", 2),
 @("for our future!", -2)
)
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
for ($i=0; $i -lt $lines.Count; $i++) {
  $txt = $lines[$i][0]
  if ($txt -eq "") { continue }
  $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
  $s.SelectVoice("Microsoft David Desktop")
  $s.Rate = $lines[$i][1]
  $s.SetOutputToWaveFile("$PSScriptRoot\build\vo$($i+1).wav", $fmt)
  $s.Speak($txt); $s.Dispose()
}
