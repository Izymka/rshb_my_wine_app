// Растеризует SVG в PNG средствами AppKit (macOS), без внешних зависимостей.
// Аргументы — четвёрки: <svg> <png> <ширина> <высота>.
import AppKit

let args = Array(CommandLine.arguments.dropFirst())
for i in stride(from: 0, to: args.count, by: 4) {
  guard let image = NSImage(contentsOf: URL(fileURLWithPath: args[i])) else { fatalError("Не читается \(args[i])") }
  let width = Int(args[i + 2])!, height = Int(args[i + 3])!
  let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
                                samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                bytesPerRow: 0, bitsPerPixel: 0)!
  NSGraphicsContext.saveGraphicsState()
  NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
  image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
  NSGraphicsContext.restoreGraphicsState()
  try! bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: args[i + 1]))
}
