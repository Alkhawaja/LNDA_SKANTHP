# Define the datasets
$datasets = @( "UK_DALE","REDD","IRISE","REFIT")

# Loop through each dataset and run the Python script

foreach ($dataset in $datasets) {
    for ($i = 0; $i -lt 3; $i++) {
        Write-Host "Dataset: $dataset, iteration: $i"
        python main.py -data $dataset -iteration $i
    }
}
