// Change the path below to the base directory
base_directory = "CHANGE";
path = base_directory + getTitle();

// Make masks
roiManager("Fill");
roiManager("XOR");
run("Create Mask");

// Save to output
saveAs("TIF", path);
close();
close();
roiManager("Delete");
