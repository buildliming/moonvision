///|
fn reference_json_reject(request : String) -> Unit raise {
  assert_true(
    @json.parse(@moonwarp.process_json(request))
    is Object({ "ok": False, "error": String(_), .. }),
  )
}

///|
test "JSON integer fields reject fractions instead of silently truncating" {
  let valid =
    #|{"action":"sample","input":{"shape":[1,1,1,1],"data":[2]},"theta":[1,0,0,0,1,0],"out_height":1,"out_width":1}
  assert_true(
    @json.parse(@moonwarp.process_json(valid)) is Object({ "ok": True, .. }),
  )
  for
    invalid in [
      valid.replace(old="[1,1,1,1]", new="[1,1,1,1.9]"),
      valid.replace(old="[1,1,1,1]", new="[1,1,1.01,1]"),
      valid.replace(old="\"out_height\":1", new="\"out_height\":1.9"),
      valid.replace(old="\"out_width\":1", new="\"out_width\":1.9"),
      (
        #|{"action":"sample","input":{"shape":[1,1,1,1],"data":[2]},"grid":{"shape":[1,1,1.9,2],"data":[0,0]}}
      ),
      (
        #|{"action":"register","input":{"shape":[1,1,1,1],"data":[2]},"target":{"shape":[1,1,1,1],"data":[2]},"iterations":0.9}
      ),
    ] {
    reference_json_reject(invalid)
  }
}

///|
test "JSON boundary rejects nonfinite numbers malformed booleans and huge shapes" {
  for
    request in [
      (
        #|{"action":"sample","input":{"shape":[1,1,1,1],"data":[1e999]},"theta":[1,0,0,0,1,0],"out_height":1,"out_width":1}
      ),
      (
        #|{"action":"sample","input":{"shape":[1,1,1,1],"data":[2]},"theta":[1,0,0,0,1,0],"out_height":1,"out_width":1,"align_corners":"false"}
      ),
      (
        #|{"action":"sample","input":{"shape":[2147483647,2147483647,2147483647,2147483647],"data":[]},"theta":[],"out_height":1,"out_width":1}
      ),
      (
        #|{"action":"sample","input":{"shape":[1,1,1,1],"data":[2]},"theta":[1,0,0,0,1,0],"out_height":4294967297,"out_width":1}
      ),
    ] {
    reference_json_reject(request)
  }
}
