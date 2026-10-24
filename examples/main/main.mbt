///|
fn main {
  let augmentation =
    #|{"action":"augment","sample":{"image":{"width":2,"height":2,"channels":1,"data":[10,20,30,40]},"mask":[0,1,0,1],"boxes":[{"x_min":1,"y_min":0,"x_max":2,"y_max":2,"label":"object"}]},"pipeline":[{"op":"hflip"}],"seed":42,"sample_id":"example"}
  println("MoonAug: " + @aug.process_json(augmentation))
  let warp =
    #|{"action":"sample","input":{"shape":[1,1,2,2],"data":[1,2,3,4]},"theta":[1,0,0,0,1,0],"out_height":2,"out_width":2,"align_corners":false}
  println("MoonWarp: " + @warp.process_json(warp))
}
