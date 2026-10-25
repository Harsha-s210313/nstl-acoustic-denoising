$docxPath = "C:\Users\HARSHA\Documents\PROJECT_REPORT.docx"
$pdfPath  = "C:\Users\HARSHA\Documents\PROJECT_REPORT.pdf"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open($docxPath)
$wdFormatPDF = 17
$doc.SaveAs([ref]$pdfPath, [ref]$wdFormatPDF)
$doc.Close()
$word.Quit()
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
Write-Host "Successfully generated PDF: $pdfPath"
