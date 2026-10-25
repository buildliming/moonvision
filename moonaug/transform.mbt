///|
fn byte_value(value : Double) -> Byte {
  @math.floor(value.max(0.0).min(255.0) + 0.5).to_int().to_byte()
}

///|
fn clip_box(
  box : BBox,
  width : Int,
  height : Int,
  min_visibility : Double,
) -> BBox? {
  let x_min = box.x_min.max(0.0).min(width.to_double())
  let y_min = box.y_min.max(0.0).min(height.to_double())
  let x_max = box.x_max.max(0.0).min(width.to_double())
  let y_max = box.y_max.max(0.0).min(height.to_double())
  let original_width = box.x_max - box.x_min
  let original_height = box.y_max - box.y_min
  let visible_width = x_max - x_min
  let visible_height = y_max - y_min
  let retained = visible_width /
    original_width *
    (visible_height / original_height)
  if visible_width > 0.0 && visible_height > 0.0 && retained >= min_visibility {
    Some({ x_min, y_min, x_max, y_max, label: box.label, })
  } else {
    None
  }
}

///|
fn keep_valid_boxes(boxes : Array[BBox]) -> Array[BBox] {
  boxes.filter(fn(box) { box.x_min < box.x_max && box.y_min < box.y_max })
}

// Map an output pixel to the corresponding input pixel, preserving class IDs exactly.

///|
fn remap(
  sample : Sample,
  width : Int,
  height : Int,
  source : (Int, Int) -> (Int, Int),
  boxes : Array[BBox],
) -> Sample {
  let channels = sample.image.channels
  let data = Array::make(width * height * channels, b'\x00')
  let mask = sample.mask.map(fn(_) { Array::make(width * height, 0) })
  for y in 0..<height {
    for x in 0..<width {
      let (sx, sy) = source(x, y)
      let src = sy * sample.image.width + sx
      let dst = y * width + x
      for c in 0..<channels {
        data[dst * channels + c] = sample.image.data[src * channels + c]
      }
      if mask is Some(out) && sample.mask is Some(input) {
        out[dst] = input[src]
      }
    }
  }
  { image: { width, height, channels, data, }, mask, boxes, }
}

///|
fn resize_sample(sample : Sample, width : Int, height : Int) -> Sample {
  let image = sample.image
  let channels = image.channels
  let data = Array::make(width * height * channels, b'\x00')
  let mask = sample.mask.map(fn(_) { Array::make(width * height, 0) })
  let ratio_x = image.width.to_double() / width.to_double()
  let ratio_y = image.height.to_double() / height.to_double()
  for y in 0..<height {
    let fy = ((y.to_double() + 0.5) * ratio_y - 0.5).max(0.0)
    let y0 = @math.floor(fy).to_int().min(image.height - 1)
    let y1 = (y0 + 1).min(image.height - 1)
    let wy = fy - y0.to_double()
    let my = @math.floor((y.to_double() + 0.5) * ratio_y)
      .to_int()
      .min(image.height - 1)
    for x in 0..<width {
      let fx = ((x.to_double() + 0.5) * ratio_x - 0.5).max(0.0)
      let x0 = @math.floor(fx).to_int().min(image.width - 1)
      let x1 = (x0 + 1).min(image.width - 1)
      let wx = fx - x0.to_double()
      for c in 0..<channels {
        let a = image.data[(y0 * image.width + x0) * channels + c]
          .to_int()
          .to_double()
        let b = image.data[(y0 * image.width + x1) * channels + c]
          .to_int()
          .to_double()
        let c0 = image.data[(y1 * image.width + x0) * channels + c]
          .to_int()
          .to_double()
        let d = image.data[(y1 * image.width + x1) * channels + c]
          .to_int()
          .to_double()
        data[(y * width + x) * channels + c] = byte_value(
          (a * (1.0 - wx) + b * wx) * (1.0 - wy) +
          (c0 * (1.0 - wx) + d * wx) * wy,
        )
      }
      if mask is Some(out) && sample.mask is Some(input) {
        let mx = @math.floor((x.to_double() + 0.5) * ratio_x)
          .to_int()
          .min(image.width - 1)
        out[y * width + x] = input[my * image.width + mx]
      }
    }
  }
  let scale_x = width.to_double() / image.width.to_double()
  let scale_y = height.to_double() / image.height.to_double()
  let boxes = keep_valid_boxes(
    sample.boxes.map(fn(box) {
      {
        ..box,
        x_min: (box.x_min * scale_x).max(0.0).min(width.to_double()),
        x_max: (box.x_max * scale_x).max(0.0).min(width.to_double()),
        y_min: (box.y_min * scale_y).max(0.0).min(height.to_double()),
        y_max: (box.y_max * scale_y).max(0.0).min(height.to_double()),
      }
    }),
  )
  { image: { width, height, channels, data, }, mask, boxes, }
}

///|
fn apply_transform(
  sample : Sample,
  transform : Transform,
  min_visibility : Double,
) -> Sample raise AugError {
  validate_transform(transform)
  let w = sample.image.width
  let h = sample.image.height
  match transform {
    Resize(width, height) => resize_sample(sample, width, height)
    Crop(x, y, width, height) => {
      guard width <= w && height <= h && x <= w - width && y <= h - height else {
        raise Invalid("crop must lie entirely inside the current image")
      }
      let boxes = []
      for box in sample.boxes {
        let moved = {
          ..box,
          x_min: box.x_min - x.to_double(),
          x_max: box.x_max - x.to_double(),
          y_min: box.y_min - y.to_double(),
          y_max: box.y_max - y.to_double(),
        }
        if clip_box(moved, width, height, min_visibility) is Some(clipped) {
          boxes.push(clipped)
        }
      }
      remap(sample, width, height, fn(dx, dy) { (dx + x, dy + y) }, boxes)
    }
    HFlip => {
      let boxes = keep_valid_boxes(
        sample.boxes.map(fn(box) {
          {
            ..box,
            x_min: w.to_double() - box.x_max,
            x_max: w.to_double() - box.x_min,
          }
        }),
      )
      remap(sample, w, h, fn(x, y) { (w - 1 - x, y) }, boxes)
    }
    VFlip => {
      let boxes = keep_valid_boxes(
        sample.boxes.map(fn(box) {
          {
            ..box,
            y_min: h.to_double() - box.y_max,
            y_max: h.to_double() - box.y_min,
          }
        }),
      )
      remap(sample, w, h, fn(x, y) { (x, h - 1 - y) }, boxes)
    }
    Rotate90(turns) => {
      let turns = (turns % 4 + 4) % 4
      let mut output = sample
      for _ in 0..<turns {
        let iw = output.image.width
        let ih = output.image.height
        let boxes = keep_valid_boxes(
          output.boxes.map(fn(box) {
            {
              x_min: ih.to_double() - box.y_max,
              y_min: box.x_min,
              x_max: ih.to_double() - box.y_min,
              y_max: box.x_max,
              label: box.label,
            }
          }),
        )
        output = remap(output, ih, iw, fn(x, y) { (y, ih - 1 - x) }, boxes)
      }
      output
    }
    Brightness(factor) | Contrast(factor) => {
      let data = sample.image.data.copy()
      let channels = sample.image.channels
      for i in 0..<data.length() {
        if channels != 4 || i % channels != 3 {
          let value = data[i].to_int().to_double()
          data[i] = match transform {
            Brightness(_) => byte_value(value * factor)
            _ => byte_value((value - 128.0) * factor + 128.0)
          }
        }
      }
      { ..sample, image: { ..sample.image, data, }, }
    }
    Pad(padding) => pad_sample(sample, padding)
    RandomCrop(_, _) | Letterbox(_) | RandomResizedCrop(_) =>
      raise Invalid("replay cannot contain unresolved high-level transforms")
  }
}

///|
fn pad_sample(sample : Sample, padding : Padding) -> Sample raise AugError {
  let image = sample.image
  let (width, height) = padded_shape(image.width, image.height, padding)
  guard padding.fill.length() == image.channels else {
    raise Invalid("padding fill length must equal image channels")
  }
  let data = Array::make(width * height * image.channels, b'\x00')
  for pixel in 0..<(width * height) {
    for channel in 0..<image.channels {
      data[pixel * image.channels + channel] = padding.fill[channel]
    }
  }
  let mask = sample.mask.map(fn(_) {
    Array::make(width * height, padding.mask_value)
  })
  for y in 0..<image.height {
    for x in 0..<image.width {
      let src = y * image.width + x
      let dst = (y + padding.top) * width + x + padding.left
      for channel in 0..<image.channels {
        data[dst * image.channels + channel] = image.data[src * image.channels +
          channel]
      }
      if mask is Some(out) && sample.mask is Some(input) {
        out[dst] = input[src]
      }
    }
  }
  let dx = padding.left.to_double()
  let dy = padding.top.to_double()
  let boxes = keep_valid_boxes(
    sample.boxes.map(fn(box) {
      {
        ..box,
        x_min: box.x_min + dx,
        x_max: box.x_max + dx,
        y_min: box.y_min + dy,
        y_max: box.y_max + dy,
      }
    }),
  )
  { image: { width, height, channels: image.channels, data, }, mask, boxes, }
}
