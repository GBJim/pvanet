from train_group import train_group

if __name__ == "__main__":
    output_dir = "models/CP_tuned"
    vatic_names = ["YuDa","A1HighwayDay", "B2HighwayNight"]
    mapper = {"van":"car", "truck":"car", "trailer-head":"car",\
              "sedan/suv":"car", "scooter":"motorbike", "bike":"bicycle"}
    
    solver = "models/pvanet/example_train/solver.prototxt"
    train_pt = "models/pvanet/example_train/train.prototxt"
    caffenet = "models/pvanet/pva9.1/PVA9.1_ImgNet_COCO_VOC0712plus.caffemodel"
    max_iters = 10000
    output_name = "CP-YAB"
    net_params = (solver, train_pt, caffenet, max_iters, output_name)
    
    
    class_set_name = "voc"
    train_group(net_params, vatic_names, class_set_name, output_dir, CLS_mapper=mapper)