import UIKit
import SceneKit
import ARKit
import AVFoundation

class ViewController: UIViewController, ARSCNViewDelegate {
    
    @IBOutlet weak var sceneView: ARSCNView!
    @IBOutlet weak var pathView: UIView!
    @IBOutlet weak var statusLabel: UILabel!
    
    private var startPoint: SCNVector3?
    private var endPoint: SCNVector3?
    private var pathNodes: [SCNNode] = []
    private var pathPoints: [SCNVector3] = []
    private var obstacles: [SCNVector3] = []
    private var isSettingEndPoint = false
    private let speechSynthesizer = AVSpeechSynthesizer()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Set up ARSCNView
        sceneView.delegate = self
        sceneView.showsStatistics = true
        sceneView.scene = SCNScene() // Empty scene
        
        // Add tap gesture for setting end point
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        sceneView.addGestureRecognizer(tapGesture)
        
        // Configure pathView UI
        pathView.layer.borderWidth = 1.0
        pathView.layer.borderColor = UIColor.black.cgColor
        pathView.alpha = 0.5 // Semi-transparent to reduce obstruction
        
        // Add Auto Layout constraints
        sceneView.translatesAutoresizingMaskIntoConstraints = false
        pathView.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            sceneView.topAnchor.constraint(equalTo: view.topAnchor),
            sceneView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            sceneView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            sceneView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            pathView.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            pathView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 10),
            pathView.widthAnchor.constraint(equalToConstant: 100),
            pathView.heightAnchor.constraint(equalToConstant: 100),
            
            statusLabel.leadingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.leadingAnchor, constant: 10),
            statusLabel.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor, constant: -10),
            statusLabel.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -10)
        ])
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Start AR session with horizontal plane detection
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = .horizontal
        sceneView.session.run(configuration)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        sceneView.session.pause()
    }
    
    // MARK: - ARSCNViewDelegate
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        guard let planeAnchor = anchor as? ARPlaneAnchor else { return }
        
        // Store plane center as an obstacle (simplified for now)
        let position = SCNVector3(planeAnchor.center.x, planeAnchor.center.y, planeAnchor.center.z)
        obstacles.append(position)
    }
    
    // MARK: - Button Actions
    
    @IBAction func setEndPointTapped(_ sender: UIButton) {
        isSettingEndPoint = true
        statusLabel.text = "Tap to set End Point"
    }
    
    @IBAction func findPathTapped(_ sender: UIButton) {
        guard let end = endPoint else {
            statusLabel.text = "Please set End Point"
            speak("Please set the end point.")
            return
        }
        
        // Get camera position as start point, projected to floor
        guard let cameraTransform = sceneView.session.currentFrame?.camera.transform else {
            statusLabel.text = "Unable to get camera position"
            speak("Unable to get camera position.")
            return
        }
        let cameraPosition = SCNVector3(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
        startPoint = SCNVector3(cameraPosition.x, end.y, cameraPosition.z) // Align with floor
        
        // Generate and display path
        pathPoints = findPath(from: startPoint!, to: end, avoiding: obstacles)
        if pathPoints.isEmpty {
            statusLabel.text = "No path found"
            speak("No path found.")
            return
        }
        
        displayPath()
        display2DPath()
        statusLabel.text = "Path found!"
        speak("Path found. Follow the path.")
        startNavigation()
    }
    
    // MARK: - Tap Gesture Handling
    
    @objc func handleTap(_ gesture: UITapGestureRecognizer) {
        guard isSettingEndPoint else { return }
        
        let location = gesture.location(in: sceneView)
        guard let query = sceneView.raycastQuery(from: location, allowing: .existingPlaneInfinite, alignment: .horizontal),
              let result = sceneView.session.raycast(query).first else {
            statusLabel.text = "Tap on a detected plane"
            speak("Please tap on a detected plane.")
            return
        }
        
        let position = SCNVector3(result.worldTransform.columns.3.x,
                                  result.worldTransform.columns.3.y,
                                  result.worldTransform.columns.3.z)
        endPoint = position
        
        // Add red end point marker
        let sphere = SCNSphere(radius: 0.02)
        sphere.firstMaterial?.diffuse.contents = UIColor.red
        let endNode = SCNNode(geometry: sphere)
        endNode.position = position
        sceneView.scene.rootNode.addChildNode(endNode)
        pathNodes.append(endNode)
        
        statusLabel.text = "End Point set"
        speak("End point set.")
        isSettingEndPoint = false
    }
    
    // MARK: - Pathfinding
    
    func findPath(from start: SCNVector3, to end: SCNVector3, avoiding obstacles: [SCNVector3]) -> [SCNVector3] {
        var path: [SCNVector3] = [start]
        let stepSize: Float = 0.1
        var currentPosition = start
        
        let dx = end.x - start.x
        let dz = end.z - start.z
        let distance = sqrt(dx * dx + dz * dz)
        
        if distance == 0 { return [start] }
        
        let normalizedDx = dx / distance
        let normalizedDz = dz / distance
        
        while distanceBetween(currentPosition, end) > stepSize {
            let nextX = currentPosition.x + normalizedDx * stepSize
            let nextZ = currentPosition.z + normalizedDz * stepSize
            let nextPosition = SCNVector3(nextX, end.y, nextZ) // Keep on floor
            
            // Simplified obstacle check
            var isSafe = true
            for obstacle in obstacles {
                if distanceBetween(nextPosition, obstacle) < 0.5 {
                    isSafe = false
                    break
                }
            }
            
            if isSafe {
                path.append(nextPosition)
                currentPosition = nextPosition
            } else {
                // Stop path if obstacle is detected
                break
            }
        }
        path.append(end)
        return path
    }
    
    func distanceBetween(_ a: SCNVector3, _ b: SCNVector3) -> Float {
        let dx = a.x - b.x
        let dy = a.y - b.y
        let dz = a.z - b.z
        return sqrt(dx * dx + dy * dy + dz * dz)
    }
    
    // MARK: - Path Display
    
    func displayPath() {
        pathNodes.forEach { $0.removeFromParentNode() }
        pathNodes.removeAll()
        
        if let end = endPoint {
            let sphere = SCNSphere(radius: 0.02)
            sphere.firstMaterial?.diffuse.contents = UIColor.red
            let endNode = SCNNode(geometry: sphere)
            endNode.position = end
            sceneView.scene.rootNode.addChildNode(endNode)
            pathNodes.append(endNode)
        }
        
        for point in pathPoints {
            let sphere = SCNSphere(radius: 0.01)
            sphere.firstMaterial?.diffuse.contents = UIColor.green
            let node = SCNNode(geometry: sphere)
            node.position = point
            sceneView.scene.rootNode.addChildNode(node)
            pathNodes.append(node)
        }
        
        for i in 0..<pathPoints.count - 1 {
            let start = pathPoints[i]
            let end = pathPoints[i + 1]
            if let lineNode = createLineNode(from: start, to: end, color: .green) {
                sceneView.scene.rootNode.addChildNode(lineNode)
                pathNodes.append(lineNode)
            }
        }
    }
    
    func createLineNode(from start: SCNVector3, to end: SCNVector3, color: UIColor) -> SCNNode? {
        let vector = SCNVector3(end.x - start.x, end.y - start.y, end.z - start.z)
        let length = sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)
        if length == 0 {
            return nil
        }
        let cylinder = SCNCylinder(radius: 0.005, height: CGFloat(length))
        cylinder.firstMaterial?.diffuse.contents = color
        let lineNode = SCNNode(geometry: cylinder)
        lineNode.position = SCNVector3(start.x + vector.x / 2, start.y + vector.y / 2, start.z + vector.z / 2)
        lineNode.eulerAngles = SCNVector3(0, acos(vector.z / length), atan2(vector.y, vector.x))
        return lineNode
    }
    
    // MARK: - 2D Path Display
    
    func display2DPath() {
        pathView.layer.sublayers?.forEach { $0.removeFromSuperlayer() }
        
        guard !pathPoints.isEmpty else { return }
        
        let bounds = pathView.bounds
        let minX = pathPoints.map { $0.x }.min() ?? 0
        let maxX = pathPoints.map { $0.x }.max() ?? 0
        let minZ = pathPoints.map { $0.z }.min() ?? 0
        let maxZ = pathPoints.map { $0.z }.max() ?? 0
        
        let widthRange = maxX - minX
        let depthRange = maxZ - minZ
        let scaleX = widthRange == 0 ? 1 : Float(bounds.width) / widthRange
        let scaleZ = depthRange == 0 ? 1 : Float(bounds.height) / depthRange
        let scale = min(scaleX, scaleZ) * 0.8
        
        let offsetX = Float(bounds.midX) - (maxX + minX) / 2 * scale
        let offsetY = Float(bounds.midY) - (maxZ + minZ) / 2 * scale
        
        let pathLayer = CAShapeLayer()
        let path = UIBezierPath()
        let firstPoint = pathPoints[0]
        let firstX = CGFloat(firstPoint.x * scale + offsetX)
        let firstY = CGFloat(firstPoint.z * scale + offsetY)
        path.move(to: CGPoint(x: firstX, y: firstY))
        
        for point in pathPoints.dropFirst() {
            let x = CGFloat(point.x * scale + offsetX)
            let y = CGFloat(point.z * scale + offsetY)
            path.addLine(to: CGPoint(x: x, y: y))
        }
        
        pathLayer.path = path.cgPath
        pathLayer.strokeColor = UIColor.green.cgColor
        pathLayer.lineWidth = 3.0
        pathLayer.fillColor = nil
        pathView.layer.addSublayer(pathLayer)
    }
    
    // MARK: - Auditory Navigation
    
    func startNavigation() {
        Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] timer in
            guard let self = self, let end = self.endPoint, !self.pathPoints.isEmpty else {
                timer.invalidate()
                return
            }
            
            guard let cameraTransform = self.sceneView.session.currentFrame?.camera.transform else { return }
            let cameraPosition = SCNVector3(cameraTransform.columns.3.x, cameraTransform.columns.3.y, cameraTransform.columns.3.z)
            
            if self.distanceBetween(cameraPosition, end) < 0.2 {
                self.statusLabel.text = "Destination reached!"
                self.speak("Destination reached.")
                timer.invalidate()
                return
            }
            
            // Find closest path point manually
            var closestPoint = self.pathPoints[0]
            var minDistance = self.distanceBetween(cameraPosition, closestPoint)
            for point in self.pathPoints {
                let distance = self.distanceBetween(cameraPosition, point)
                if distance < minDistance {
                    minDistance = distance
                    closestPoint = point
                }
            }
            
            // Find the index of the closest point
            var closestIndex = 0
            for (index, point) in self.pathPoints.enumerated() {
                if point.x == closestPoint.x && point.y == closestPoint.y && point.z == closestPoint.z {
                    closestIndex = index
                    break
                }
            }
            
            if closestIndex < self.pathPoints.count - 1 {
                let nextPoint = self.pathPoints[closestIndex + 1]
                let dx = nextPoint.x - cameraPosition.x
                let dz = nextPoint.z - cameraPosition.z
                let direction = SCNVector3(dx, 0, dz)
                
                let forwardX = -cameraTransform.columns.2.x
                let forwardZ = -cameraTransform.columns.2.z
                let cameraForward = SCNVector3(forwardX, 0, forwardZ)
                
                let angle = atan2(direction.x, direction.z) - atan2(cameraForward.x, cameraForward.z)
                
                if abs(angle) < .pi / 4 {
                    self.speak("Move forward.")
                } else if angle > 0 {
                    self.speak("Turn right.")
                } else {
                    self.speak("Turn left.")
                }
            }
        }
    }
    
    func speak(_ message: String) {
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        utterance.rate = 0.5
        speechSynthesizer.speak(utterance)
    }
}
