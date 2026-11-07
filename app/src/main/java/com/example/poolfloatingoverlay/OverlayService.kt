package com.example.poolfloatingoverlay

import android.app.Activity
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.content.res.Configuration
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.PixelFormat
import android.graphics.PointF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.RectF
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.os.Looper
import android.os.SystemClock
import android.view.Display
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.Switch
import android.widget.TextView
import java.nio.ByteBuffer
import java.util.Arrays
import org.opencv.android.OpenCVLoader
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Optional OpenCV circle detection.
 *
 * The Maven AAR ships prebuilt .so files, so nothing is compiled here. If the native library
 * cannot load, the Kotlin detector keeps working on its own.
 */
private object CvBalls {
    var ready = false
        private set

    fun init(): Boolean {
        ready = try {
            OpenCVLoader.initLocal()
        } catch (error: Throwable) {
            false
        }
        return ready
    }

    /** Returns circles as [x, y, radius] in frame pixels, or null when OpenCV is unavailable. */
    fun detect(image: Image, maxRadius: Int, sensitivity: Int): List<FloatArray>? {
        if (!ready || maxRadius <= 0) return null
        var rgba: Mat? = null
        var view: Mat? = null
        var small: Mat? = null
        var gray: Mat? = null
        var circles: Mat? = null
        try {
            val plane = image.planes[0]
            val buffer = plane.buffer
            val rowStride = plane.rowStride
            val width = image.width
            val height = image.height
            if (width <= 0 || height <= 0 || rowStride < width * 4) return null

            val bytes = ByteArray(rowStride * height)
            buffer.position(0)
            buffer.get(bytes, 0, minOf(bytes.size, buffer.remaining()))

            val full = Mat(height, rowStride / 4, CvType.CV_8UC4)
            full.put(0, 0, bytes)
            rgba = full
            val cropped = full.submat(0, height, 0, width)
            view = cropped

            val scale = 3.0
            val reduced = Mat()
            Imgproc.resize(cropped, reduced, Size(width / scale, height / scale))
            small = reduced
            val mono = Mat()
            Imgproc.cvtColor(reduced, mono, Imgproc.COLOR_RGBA2GRAY)
            gray = mono
            Imgproc.GaussianBlur(mono, mono, Size(5.0, 5.0), 1.5)

            val found = Mat()
            circles = found
            Imgproc.HoughCircles(
                mono,
                found,
                Imgproc.HOUGH_GRADIENT,
                1.0,
                (maxRadius / scale) * 1.4,
                90.0,
                sensitivity.toDouble(),
                ((maxRadius * 0.3) / scale).toInt().coerceAtLeast(4),
                (maxRadius / scale).toInt().coerceAtLeast(6)
            )

            val out = ArrayList<FloatArray>()
            for (index in 0 until found.cols()) {
                val data = found.get(0, index) ?: continue
                if (data.size < 3) continue
                out.add(
                    floatArrayOf(
                        (data[0] * scale).toFloat(),
                        (data[1] * scale).toFloat(),
                        (data[2] * scale).toFloat()
                    )
                )
            }
            return out
        } catch (error: Throwable) {
            return null
        } finally {
            small?.release()
            gray?.release()
            circles?.release()
            view?.release()
            rgba?.release()
        }
    }
}

/**
 * Floating aim assistant with real-time screen analysis.
 *
 * Frames come from MediaProjection + ImageReader and are processed with plain Kotlin pixel maths
 * (no OpenCV, no .so libraries), so the project builds with the Android SDK alone.
 */
class OverlayService : Service() {

    companion object {
        const val EXTRA_RESULT_CODE = "RESULT_CODE"
        const val EXTRA_DATA_INTENT = "DATA_INTENT"
        const val ACTION_STOP = "com.example.poolfloatingoverlay.STOP"
        private const val CHANNEL_ID = "pool_aim_overlay"
        private const val NOTIFICATION_ID = 8108
        private const val FRAME_INTERVAL_MS = 90L
    }

    private lateinit var windowManager: WindowManager
    private lateinit var displayManager: DisplayManager
    private var bubble: View? = null
    private var panel: View? = null
    private var aimView: AimCanvasView? = null
    private var statusLabel: TextView? = null
    private var modeChip: TextView? = null
    private var showTarget = true
    private var showCushion = true

    private var projection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var captureThread: HandlerThread? = null
    private var captureHandler: Handler? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    private var resultCode = Activity.RESULT_CANCELED
    private var resultData: Intent? = null
    private var detector: BallDetector? = null
    private var frameWidth = 0
    private var frameHeight = 0
    private var lastFrameAt = 0L
    private var aimEnabled = true
    private var lastStatus = ""
    private var lastMode = "SEARCH"
    private var tuning = BallDetector.Tuning()
    private var useOpenCv = false

    private val projectionCallback = object : MediaProjection.Callback() {
        override fun onStop() {
            releaseCapture()
        }
    }

    /** A rotation changes the mirrored frame size, so the virtual display has to follow it. */
    private val displayListener = object : DisplayManager.DisplayListener {
        override fun onDisplayAdded(displayId: Int) {}

        override fun onDisplayRemoved(displayId: Int) {}

        override fun onDisplayChanged(displayId: Int) {
            if (displayId != Display.DEFAULT_DISPLAY) return
            if (projection == null) return
            mainHandler.post { rebuildDisplay() }
        }
    }

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        displayManager = getSystemService(Context.DISPLAY_SERVICE) as DisplayManager
        displayManager.registerDisplayListener(displayListener, mainHandler)
        useOpenCv = CvBalls.init()
        startForegroundCompat()
        addAimLayer()
        addBubble()
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        if (projection != null) mainHandler.post { rebuildDisplay() }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stopSelf()
            return START_NOT_STICKY
        }
        startForegroundCompat()

        val code = intent?.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
            ?: Activity.RESULT_CANCELED
        val data: Intent? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent?.getParcelableExtra(EXTRA_DATA_INTENT, Intent::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent?.getParcelableExtra(EXTRA_DATA_INTENT)
        }
        if (code == Activity.RESULT_OK && data != null) {
            resultCode = code
            resultData = data
            startCapture()
        }
        return START_STICKY
    }

    // region foreground service

    private fun buildNotification(): Notification {
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(CHANNEL_ID, "Pool aim overlay", NotificationManager.IMPORTANCE_LOW)
            channel.description = "Keeps the aim overlay and screen analysis running"
            manager.createNotificationChannel(channel)
        }
        val open = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }
        return builder
            .setContentTitle("Pool Aim Assistant")
            .setContentText("Analysing the screen in real time")
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .setOngoing(true)
            .setContentIntent(open)
            .build()
    }

    private fun startForegroundCompat() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    // endregion

    // region screen capture

    private fun startCapture() {
        if (projection != null) return
        val data = resultData ?: return
        val manager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        val created = try {
            manager.getMediaProjection(resultCode, data)
        } catch (error: Exception) {
            null
        } ?: return

        projection = created
        // Android 14 requires the callback before any virtual display exists.
        created.registerCallback(projectionCallback, mainHandler)

        if (captureThread == null) {
            val thread = HandlerThread("pool-capture")
            thread.start()
            captureThread = thread
            captureHandler = Handler(thread.looper)
        }
        createDisplay()
    }

    /** (Re)creates the mirror at the current display size; also used after a rotation. */
    private fun createDisplay() {
        val active = projection ?: return
        val handler = captureHandler ?: return
        teardownDisplay()

        val metrics = resources.displayMetrics
        val width = metrics.widthPixels
        val height = metrics.heightPixels
        if (width <= 0 || height <= 0) return

        val reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 3)
        imageReader = reader
        frameWidth = width
        frameHeight = height
        detector = BallDetector(width, height)
        virtualDisplay = active.createVirtualDisplay(
            "pool-capture",
            width,
            height,
            metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader.surface,
            null,
            handler
        )
        reader.setOnImageAvailableListener({ source -> onFrameAvailable(source) }, handler)
        aimView?.setFrameSize(width, height)
        updateStatus("capture ${width}x$height")
    }

    private fun teardownDisplay() {
        val reader = imageReader
        val display = virtualDisplay
        imageReader = null
        virtualDisplay = null
        detector = null
        try {
            reader?.setOnImageAvailableListener(null, null)
        } catch (error: Exception) {
        }
        try {
            display?.release()
        } catch (error: Exception) {
        }
        try {
            reader?.close()
        } catch (error: Exception) {
        }
        aimView?.clearScene()
    }

    private fun rebuildDisplay() {
        if (projection == null) return
        val metrics = resources.displayMetrics
        if (metrics.widthPixels == frameWidth && metrics.heightPixels == frameHeight && imageReader != null) return
        createDisplay()
    }

    private fun onFrameAvailable(source: ImageReader) {
        val image: Image = try {
            source.acquireLatestImage()
        } catch (error: Exception) {
            null
        } ?: return

        try {
            val now = SystemClock.elapsedRealtime()
            if (now - lastFrameAt < FRAME_INTERVAL_MS) return
            lastFrameAt = now
            if (!aimEnabled) return
            val scene = detector?.analyse(image, tuning, useOpenCv) ?: return
            val width = frameWidth
            val height = frameHeight
            mainHandler.post {
                aimView?.showScene(scene, width, height)
                updateStatus(scene.status(), scene.mode())
            }
        } catch (error: Throwable) {
            // A single broken frame must never stop the capture thread.
        } finally {
            image.close()
        }
    }

    private fun releaseCapture() {
        teardownDisplay()
        val active = projection
        projection = null
        resultData = null
        try {
            active?.unregisterCallback(projectionCallback)
        } catch (error: Exception) {
        }
        try {
            active?.stop()
        } catch (error: Exception) {
        }
        captureThread?.quitSafely()
        captureThread = null
        captureHandler = null
        aimView?.clearScene()
        updateStatus("capture stopped")
    }

    // endregion

    // region overlay views

    private fun overlayParams(
        width: Int,
        height: Int,
        touchable: Boolean,
        posX: Int,
        posY: Int,
        secure: Boolean
    ): WindowManager.LayoutParams {
        var flags = WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN
        if (!touchable) flags = flags or WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
        // Secure windows are left out of the mirrored frame, so the detector never sees the
        // bubble or the menu it draws itself.
        if (secure) flags = flags or WindowManager.LayoutParams.FLAG_SECURE
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
        } else {
            @Suppress("DEPRECATION")
            WindowManager.LayoutParams.TYPE_PHONE
        }
        return WindowManager.LayoutParams(width, height, type, flags, PixelFormat.TRANSLUCENT).apply {
            gravity = Gravity.TOP or Gravity.START
            this.x = posX
            this.y = posY
        }
    }

    private fun addAimLayer() {
        if (aimView != null) return
        val view = AimCanvasView(this)
        aimView = view
        if (frameWidth > 0 && frameHeight > 0) view.setFrameSize(frameWidth, frameHeight)
        addViewSafe(
            view,
            overlayParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                touchable = false,
                posX = 0,
                posY = 0,
                secure = false
            )
        )
    }

    private fun addBubble() {
        if (bubble != null) return
        val button = TextView(this).apply {
            text = "8"
            textSize = 18f
            setTextColor(Color.WHITE)
            gravity = Gravity.CENTER
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.parseColor("#0E1117"))
                setStroke(dp(2), Color.parseColor("#E8EEF6"))
            }
            elevation = 14f
        }
        val params = overlayParams(
            dp(54),
            dp(54),
            touchable = true,
            posX = dp(20),
            posY = dp(190),
            secure = true
        )
        button.setOnTouchListener(object : View.OnTouchListener {
            var downX = 0f
            var downY = 0f
            var startX = 0
            var startY = 0
            var moved = false

            override fun onTouch(view: View, event: MotionEvent): Boolean {
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        downX = event.rawX
                        downY = event.rawY
                        startX = params.x
                        startY = params.y
                        moved = false
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        val dx = (event.rawX - downX).toInt()
                        val dy = (event.rawY - downY).toInt()
                        if (abs(dx) > 5 || abs(dy) > 5) moved = true
                        params.x = startX + dx
                        params.y = startY + dy
                        try {
                            windowManager.updateViewLayout(button, params)
                        } catch (error: Exception) {
                        }
                        return true
                    }
                    MotionEvent.ACTION_UP -> {
                        if (!moved) togglePanel()
                        return true
                    }
                }
                return false
            }
        })
        bubble = button
        addViewSafe(button, params)
    }

    private fun togglePanel() {
        val current = panel
        if (current != null) {
            panel = null
            statusLabel = null
            modeChip = null
            removeViewSafe(current)
            return
        }
        val card = card()
        card.addView(panelHeader())
        card.addView(panelStatus())
        card.addView(divider())
        card.addView(toggleRow("Aim line", aimEnabled) { checked ->
            aimEnabled = checked
            aimView?.setLayers(aimLayers())
            if (!checked) aimView?.clearScene()
        })
        card.addView(toggleRow("Target ring", showTarget) { checked ->
            showTarget = checked
            aimView?.setLayers(aimLayers())
        })
        card.addView(toggleRow("Cushion path", showCushion) { checked ->
            showCushion = checked
            aimView?.setLayers(aimLayers())
        })
        card.addView(divider())
        card.addView(sectionLabel("DETECTION"))
        addSlider(card, "White brightness", 120, 240, tuning.whiteFloor) { value ->
            tuning = tuning.copy(whiteFloor = value)
        }
        addSlider(card, "White colour spread", 8, 90, tuning.whiteSpread) { value ->
            tuning = tuning.copy(whiteSpread = value)
        }
        addSlider(card, "Ball colour spread", 20, 130, tuning.colourSpread) { value ->
            tuning = tuning.copy(colourSpread = value)
        }
        addSlider(card, "OpenCV sensitivity", 12, 70, tuning.circleSensitivity) { value ->
            tuning = tuning.copy(circleSensitivity = value)
        }
        card.addView(divider())
        card.addView(panelActions())
        card.addView(link("Recapture screen size") { createDisplay() })
        card.addView(link("Stop screen capture") { releaseCapture() })

        card.layoutParams = android.widget.FrameLayout.LayoutParams(
            android.widget.FrameLayout.LayoutParams.MATCH_PARENT,
            android.widget.FrameLayout.LayoutParams.WRAP_CONTENT
        )
        val scroller = ScrollView(this).apply { addView(card) }
        panel = scroller
        addViewSafe(
            scroller,
            overlayParams(
                dp(300),
                WindowManager.LayoutParams.WRAP_CONTENT,
                touchable = true,
                posX = dp(20),
                posY = dp(170),
                secure = true
            )
        )
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private fun aimLayers(): AimCanvasView.Layers =
        AimCanvasView.Layers(aim = aimEnabled, target = showTarget, cushion = showCushion)

    private fun card(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(dp(18), dp(16), dp(18), dp(16))
        background = GradientDrawable().apply {
            cornerRadius = dp(22).toFloat()
            setColor(Color.parseColor("#F2141820"))
            setStroke(dp(1), Color.parseColor("#33FFFFFF"))
        }
        elevation = 20f
    }

    private fun panelHeader(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val title = TextView(this).apply {
            text = "POOL AIM"
            textSize = 16f
            setTextColor(Color.WHITE)
            letterSpacing = 0.12f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val chip = TextView(this).apply {
            text = lastMode
            textSize = 10f
            setTextColor(Color.parseColor("#0B1220"))
            gravity = Gravity.CENTER
            setPadding(dp(10), dp(4), dp(10), dp(4))
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
            background = GradientDrawable().apply {
                cornerRadius = dp(9).toFloat()
                setColor(Color.parseColor("#7DD3FC"))
            }
        }
        modeChip = chip
        row.addView(title, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(chip)
        return row
    }

    private fun panelStatus(): TextView = TextView(this).apply {
        text = lastStatus.ifEmpty { "waiting for frames" }
        textSize = 11.5f
        setTextColor(Color.parseColor("#8FA3B8"))
        setPadding(0, dp(6), 0, 0)
    }.also { statusLabel = it }

    private fun divider(): View = View(this).apply {
        setBackgroundColor(Color.parseColor("#1FFFFFFF"))
        layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(1)).apply {
            topMargin = dp(10)
            bottomMargin = dp(4)
        }
    }

    private fun sectionLabel(label: String): TextView = TextView(this).apply {
        text = label
        textSize = 10f
        setTextColor(Color.parseColor("#7B8CA3"))
        letterSpacing = 0.18f
        setPadding(0, dp(8), 0, dp(2))
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
    }

    private fun toggleRow(label: String, checked: Boolean, onChange: (Boolean) -> Unit): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val name = TextView(this).apply {
            text = label
            textSize = 14f
            setTextColor(Color.parseColor("#E6EDF5"))
        }
        val toggle = Switch(this).apply {
            isChecked = checked
            showText = false
            setOnCheckedChangeListener { _, value -> onChange(value) }
        }
        row.addView(name, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(toggle)
        return row
    }

    private fun pill(label: String, accent: Boolean, action: () -> Unit): TextView = TextView(this).apply {
        text = label
        textSize = 13f
        gravity = Gravity.CENTER
        setTextColor(if (accent) Color.parseColor("#0B1220") else Color.parseColor("#DCE7F2"))
        setPadding(dp(14), dp(10), dp(14), dp(10))
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        background = GradientDrawable().apply {
            cornerRadius = dp(14).toFloat()
            if (accent) {
                setColor(Color.parseColor("#7DD3FC"))
            } else {
                setColor(Color.parseColor("#1E2733"))
                setStroke(dp(1), Color.parseColor("#33414F"))
            }
        }
        setOnClickListener { action() }
    }

    private fun panelActions(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(10), 0, 0)
        }
        val engine = pill(if (useOpenCv) "OpenCV" else "Kotlin", accent = false) {
            useOpenCv = if (useOpenCv) false else CvBalls.ready
            togglePanel()
            togglePanel()
        }
        val close = pill("Close", accent = true) { togglePanel() }
        row.addView(
            engine,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply {
                rightMargin = dp(8)
            }
        )
        row.addView(close)
        return row
    }

    private fun link(label: String, action: () -> Unit): TextView = TextView(this).apply {
        text = label
        textSize = 12.5f
        setTextColor(Color.parseColor("#7DD3FC"))
        setPadding(0, dp(10), 0, 0)
        setOnClickListener { action() }
    }

    private fun addSlider(
        parent: LinearLayout,
        label: String,
        min: Int,
        max: Int,
        value: Int,
        onChange: (Int) -> Unit
    ) {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, dp(8), 0, 0)
        }
        val caption = TextView(this).apply {
            text = label
            textSize = 12.5f
            setTextColor(Color.parseColor("#B6C6D8"))
        }
        val number = TextView(this).apply {
            text = value.toString()
            textSize = 12.5f
            setTextColor(Color.WHITE)
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        row.addView(caption, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(number)
        val bar = SeekBar(this).apply {
            this.max = max - min
            progress = (value - min).coerceIn(0, max - min)
            setPadding(0, 0, 0, 0)
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    val current = min + progress
                    number.text = current.toString()
                    onChange(current)
                }

                override fun onStartTrackingTouch(seekBar: SeekBar?) {}

                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        parent.addView(row)
        parent.addView(bar)
    }

    private fun addViewSafe(view: View, params: WindowManager.LayoutParams) {
        try {
            windowManager.addView(view, params)
        } catch (error: Exception) {
            // The overlay permission can be revoked while the service runs.
        }
    }

    private fun removeViewSafe(view: View) {
        try {
            windowManager.removeView(view)
        } catch (error: Exception) {
        }
    }

    private fun updateStatus(text: String, mode: String = lastMode) {
        lastStatus = text
        lastMode = mode
        mainHandler.post {
            statusLabel?.text = text
            modeChip?.text = mode
        }
    }

    override fun onDestroy() {
        try {
            displayManager.unregisterDisplayListener(displayListener)
        } catch (error: Exception) {
        }
        releaseCapture()
        panel?.let { removeViewSafe(it) }
        bubble?.let { removeViewSafe(it) }
        aimView?.let { removeViewSafe(it) }
        panel = null
        bubble = null
        aimView = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    // endregion
}

/**
 * Draws the whole prediction: rings on the balls, the cue line, the ghost ball, the object ball
 * path and both cushion-reflected continuations. Detector coordinates are frame pixels, so they
 * are scaled to the view, which keeps every line correct after a rotation or a resolution change.
 */
class AimCanvasView(context: Context) : View(context) {

    private val density = resources.displayMetrics.density

    private val cueLinePaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 5f * density
        isAntiAlias = true
    }
    private val ghostPaint = Paint().apply {
        color = Color.rgb(0, 229, 255)
        strokeWidth = 3.5f * density
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val objectLinePaint = Paint().apply {
        color = Color.rgb(255, 64, 72)
        strokeWidth = 4.5f * density
        isAntiAlias = true
    }
    private val cuePathPaint = Paint().apply {
        color = Color.rgb(186, 104, 255)
        strokeWidth = 3.5f * density
        isAntiAlias = true
        pathEffect = DashPathEffect(floatArrayOf(18f * density, 12f * density), 0f)
    }
    private val cueRingPaint = Paint().apply {
        color = Color.argb(230, 255, 255, 255)
        strokeWidth = 2.5f * density
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val ballRingPaint = Paint().apply {
        color = Color.argb(140, 80, 255, 140)
        strokeWidth = 2f * density
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val hitRingPaint = Paint().apply {
        color = Color.rgb(74, 222, 128)
        strokeWidth = 3f * density
        style = Paint.Style.STROKE
        isAntiAlias = true
    }
    private val targetFillPaint = Paint().apply {
        color = Color.argb(46, 74, 222, 128)
        isAntiAlias = true
    }
    private val pocketPaint = Paint().apply {
        color = Color.argb(210, 255, 214, 102)
        isAntiAlias = true
    }
    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 11f * density
        isAntiAlias = true
        textAlign = Paint.Align.CENTER
        typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        setShadowLayer(3f * density, 0f, density, Color.BLACK)
    }

    /** Which parts of the prediction are drawn; the menu switches these live. */
    data class Layers(
        val aim: Boolean = true,
        val target: Boolean = true,
        val cushion: Boolean = true
    )

    private var scene: BallDetector.Scene? = null
    private var frameWidth = 0
    private var frameHeight = 0
    private var layers = Layers()

    private val path = ArrayList<PointF>(8)

    fun setFrameSize(width: Int, height: Int) {
        frameWidth = width
        frameHeight = height
        invalidate()
    }

    fun showScene(result: BallDetector.Scene, width: Int, height: Int) {
        scene = result
        frameWidth = width
        frameHeight = height
        invalidate()
    }

    fun clearScene() {
        scene = null
        invalidate()
    }

    fun setLayers(value: Layers) {
        layers = value
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val result = scene ?: return
        if (frameWidth <= 0 || frameHeight <= 0 || width <= 0 || height <= 0) return

        val scaleX = width.toFloat() / frameWidth.toFloat()
        val scaleY = height.toFloat() / frameHeight.toFloat()
        val scale = (scaleX + scaleY) / 2f

        val table = RectF(
            result.table.left * scaleX,
            result.table.top * scaleY,
            result.table.right * scaleX,
            result.table.bottom * scaleY
        )

        val cue = result.cue ?: return
        val cueX = cue.x * scaleX
        val cueY = cue.y * scaleY
        val cueRadius = (cue.radius * scale).coerceIn(6f, 60f)
        canvas.drawCircle(cueX, cueY, cueRadius, cueRingPaint)
        canvas.drawCircle(cueX, cueY, cueRadius * 0.45f, cueRingPaint)

        if (!layers.aim) return

        // Nothing is drawn until there is a confident aim; noise would paint the whole screen.
        val aim = result.aim ?: return
        val maxLength = hypot(table.width(), table.height()) * 1.4f
        val hit = result.hit
        if (hit == null) {
            if (!result.fromLine) return
            val reach = maxLength * 0.5f
            val tipX = cueX + aim.x * reach
            val tipY = cueY + aim.y * reach
            canvas.drawLine(cueX, cueY, tipX, tipY, cuePathPaint)
            canvas.drawCircle(tipX, tipY, 5f * density, ghostPaint)
            return
        }

        val hitX = hit.x * scaleX
        val hitY = hit.y * scaleY
        val hitRadius = (hit.radius * scale).coerceIn(6f, 70f)
        if (layers.target) {
            canvas.drawCircle(hitX, hitY, hitRadius, targetFillPaint)
            canvas.drawCircle(hitX, hitY, hitRadius + 3f * density, hitRingPaint)
            canvas.drawText("TARGET", hitX, hitY - hitRadius - 10f * density, labelPaint)
        } else {
            canvas.drawCircle(hitX, hitY, hitRadius, hitRingPaint)
        }

        val dx = hitX - cueX
        val dy = hitY - cueY
        val distance = hypot(dx, dy)
        if (distance < cueRadius + hitRadius) return

        val ux = dx / distance
        val uy = dy / distance

        // Ghost ball: where the cue ball has to arrive to send the target along the line of centres.
        val ghostX = hitX - ux * hitRadius * 2f
        val ghostY = hitY - uy * hitRadius * 2f
        canvas.drawLine(cueX, cueY, ghostX, ghostY, cueLinePaint)
        canvas.drawCircle(ghostX, ghostY, hitRadius, ghostPaint)

        if (layers.cushion) {
            val pocket = if (result.tableFound) potPocket(hitX, hitY, ux, uy, table) else null
            if (pocket != null) {
                canvas.drawLine(hitX, hitY, pocket.x, pocket.y, objectLinePaint)
                canvas.drawCircle(pocket.x, pocket.y, hitRadius * 0.85f, pocketPaint)
            } else {
                path.clear()
                tracePath(
                    hitX,
                    hitY,
                    ux,
                    uy,
                    table,
                    if (result.tableFound) 1 else 0,
                    maxLength * 0.35f,
                    path
                )
                drawPath(canvas, path, objectLinePaint)
            }
        }

        // The cue ball keeps travelling along the aim line after contact.
        path.clear()
        tracePath(
            cueX,
            cueY,
            ux,
            uy,
            table,
            if (result.tableFound) 1 else 0,
            maxLength * 0.35f,
            path
        )
        drawPath(canvas, path, cuePathPaint)
    }

    /** The pocket the object ball would drop into, or null when it is not lined up with one. */
    private fun potPocket(ballX: Float, ballY: Float, ux: Float, uy: Float, table: RectF): PointF? {
        if (table.width() <= 1f || table.height() <= 1f) return null
        val midX = (table.left + table.right) / 2f
        val points = arrayOf(
            PointF(table.left, table.top),
            PointF(midX, table.top),
            PointF(table.right, table.top),
            PointF(table.left, table.bottom),
            PointF(midX, table.bottom),
            PointF(table.right, table.bottom)
        )
        var best: PointF? = null
        var bestDot = 0.94f
        for (point in points) {
            val px = point.x - ballX
            val py = point.y - ballY
            val length = hypot(px, py)
            if (length < 1f) continue
            val dot = (px / length) * ux + (py / length) * uy
            if (dot > bestDot) {
                bestDot = dot
                best = point
            }
        }
        return best
    }

    private fun drawPath(canvas: Canvas, points: List<PointF>, paint: Paint) {
        for (index in 0 until points.size - 1) {
            val a = points[index]
            val b = points[index + 1]
            canvas.drawLine(a.x, a.y, b.x, b.y, paint)
        }
    }

    /** Reflects a straight path off the table rectangle for the given number of cushions. */
    private fun tracePath(
        startX: Float,
        startY: Float,
        dirX: Float,
        dirY: Float,
        table: RectF,
        bounces: Int,
        maxLength: Float,
        out: MutableList<PointF>
    ) {
        if (table.width() <= 1f || table.height() <= 1f) return
        var x = startX.coerceIn(table.left, table.right)
        var y = startY.coerceIn(table.top, table.bottom)
        var dx = dirX
        var dy = dirY
        var travelled = 0f
        out.add(PointF(x, y))
        var bouncesLeft = bounces
        while (bouncesLeft > 0) {
            var bestT = Float.MAX_VALUE
            var hitVertical = false
            if (dx > 0.0001f) {
                val t = (table.right - x) / dx
                if (t > 0.0001f && t < bestT) {
                    bestT = t
                    hitVertical = true
                }
            } else if (dx < -0.0001f) {
                val t = (table.left - x) / dx
                if (t > 0.0001f && t < bestT) {
                    bestT = t
                    hitVertical = true
                }
            }
            if (dy > 0.0001f) {
                val t = (table.bottom - y) / dy
                if (t > 0.0001f && t < bestT) {
                    bestT = t
                    hitVertical = false
                }
            } else if (dy < -0.0001f) {
                val t = (table.top - y) / dy
                if (t > 0.0001f && t < bestT) {
                    bestT = t
                    hitVertical = false
                }
            }
            if (bestT == Float.MAX_VALUE) break
            if (travelled + bestT > maxLength) {
                val clamped = maxLength - travelled
                out.add(PointF(x + dx * clamped, y + dy * clamped))
                break
            }
            travelled += bestT
            x += dx * bestT
            y += dy * bestT
            out.add(PointF(x, y))
            if (hitVertical) dx = -dx else dy = -dy
            bouncesLeft--
        }
    }
}

/**
 * Pure Kotlin table and ball detection on RGBA_8888 screen frames.
 *
 * The frame is sampled onto a coarse grid, the most common colour there is taken to be the cloth,
 * and pixels are labelled white or ball-coloured by how far they sit from that cloth colour.
 * Connected components are then filtered by size, squareness and fill, which keeps white text,
 * thin cue sticks, rails and UI panels out. The biggest component bounds the table, the cue ball is
 * the strongest white blob on it and every other round blob is a playable ball. No native code.
 */
class BallDetector(private val frameWidth: Int, private val frameHeight: Int) {

    /** Detection thresholds, changeable from the live tuning window. */
    data class Tuning(
        val whiteFloor: Int = 185,
        val whiteSpread: Int = 35,
        val colourSpread: Int = 45,
        val clothDistance: Int = 120,
        val minBallPercent: Float = 0.010f,
        val maxBallPercent: Float = 0.090f,
        val circleSensitivity: Int = 34
    )

    data class Ball(val x: Float, val y: Float, val radius: Float, val white: Boolean)

    data class Scene(
        val cue: Ball?,
        val balls: List<Ball>,
        val table: RectF,
        val tableFound: Boolean,
        val hit: Ball?,
        val secondary: Ball?,
        val aim: PointF?,
        val fromLine: Boolean
    ) {
        /** Short label for the menu chip: how the current aim was found. */
        fun mode(): String = when {
            cue == null -> "SEARCH"
            aim == null -> "NEAREST"
            fromLine -> "AIM LINE"
            else -> "STICK"
        }

        fun status(): String {
            if (cue == null) return "cue ball not found · lower the brightness or raise contrast"
            if (aim == null) return balls.size.toString() + " balls · no aim line yet, drag to aim"
            if (hit == null) return balls.size.toString() + " balls · aim line, no ball on it"
            if (fromLine) return balls.size.toString() + " balls · following the game aim line"
            return balls.size.toString() + " balls · stick direction locked"
        }
    }

    private val step = 4
    private val columns = (frameWidth + step - 1) / step
    private val rows = (frameHeight + step - 1) / step
    private val cells = IntArray(columns * rows)
    private val labels = IntArray(columns * rows)
    private val stack = IntArray(columns * rows)

    private var smoothCue: PointF? = null
    private var smoothHit: PointF? = null
    private var smoothAim: PointF? = null
    private var pendingAim: PointF? = null
    private var pendingAimFrames = 0
    private var lastTarget: PointF? = null

    private class Component(
        val kind: Int,
        val cells: Int,
        val sumX: Float,
        val sumY: Float,
        val meanSquareX: Float,
        val meanSquareXY: Float,
        val meanSquareY: Float,
        val minX: Int,
        val minY: Int,
        val maxX: Int,
        val maxY: Int
    ) {
        fun centreX(step: Int): Float = (sumX + 0.5f) * step
        fun centreY(step: Int): Float = (sumY + 0.5f) * step
        fun widthPx(step: Int): Float = (maxX - minX + 1) * step.toFloat()
        fun heightPx(step: Int): Float = (maxY - minY + 1) * step.toFloat()
        fun diameter(step: Int): Float = (widthPx(step) + heightPx(step)) / 2f
        fun fill(): Float {
            val area = ((maxX - minX + 1) * (maxY - minY + 1)).toFloat()
            return if (area <= 0f) 0f else cells / area
        }
        fun roundEnough(): Boolean {
            val w = (maxX - minX + 1).toFloat()
            val h = (maxY - minY + 1).toFloat()
            val ratio = w / h
            return ratio in 0.55f..1.8f && fill() >= 0.42f
        }
    }

    fun analyse(image: Image, tuning: Tuning, useCv: Boolean): Scene? {
        val plane = image.planes[0]
        val buffer = plane.buffer
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride
        val width = minOf(image.width, frameWidth)
        val height = minOf(image.height, frameHeight)
        if (width <= 0 || height <= 0 || rowStride <= 0 || pixelStride <= 0) return null
        val limit = buffer.limit()

        Arrays.fill(cells, 0)
        Arrays.fill(labels, 0)

        val cloth = dominantColour(buffer, rowStride, pixelStride, limit, width, height)
        labelCells(buffer, rowStride, pixelStride, limit, width, height, cloth, tuning)
        val components = collectComponents()

        val minSide = minOf(frameWidth, frameHeight).toFloat()
        val minDiameter = minSide * tuning.minBallPercent
        val maxDiameter = minSide * tuning.maxBallPercent

        var tableComponent: Component? = null
        var tableFound = false
        var table = RectF(0f, 0f, frameWidth.toFloat(), frameHeight.toFloat())
        val biggest = components.maxByOrNull { it.cells }
        if (biggest != null) {
            val area = biggest.widthPx(step) * biggest.heightPx(step)
            if (area > frameWidth * frameHeight * 0.25f) {
                tableComponent = biggest
                tableFound = true
                table = RectF(
                    biggest.minX * step.toFloat(),
                    biggest.minY * step.toFloat(),
                    (biggest.maxX + 1) * step.toFloat(),
                    (biggest.maxY + 1) * step.toFloat()
                )
            }
        }

        val heuristic = ArrayList<Ball>()
        for (component in components) {
            if (component === tableComponent) continue
            val diameter = component.diameter(step)
            if (diameter < minDiameter || diameter > maxDiameter) continue
            if (!component.roundEnough()) continue
            val cx = component.centreX(step)
            val cy = component.centreY(step)
            if (!table.contains(cx, cy)) continue
            heuristic.add(Ball(cx, cy, diameter * 0.31f, component.kind == 1))
        }

        var balls: List<Ball> = heuristic
        if (useCv) {
            val found = CvBalls.detect(image, maxDiameter.toInt(), tuning.circleSensitivity)
            if (found != null && found.isNotEmpty()) {
                val detected = ArrayList<Ball>()
                for (circle in found) {
                    val cx = circle[0]
                    val cy = circle[1]
                    val radius = circle[2]
                    if (!table.contains(cx, cy)) continue
                    if (radius * 2f < minDiameter || radius * 2f > maxDiameter) continue
                    val white = samplesWhite(buffer, rowStride, pixelStride, limit, cx.toInt(), cy.toInt())
                    detected.add(Ball(cx, cy, radius * 0.92f, white))
                }
                if (detected.isNotEmpty()) balls = detected
            }
        }

        val cueBall = balls.filter { it.white }.maxByOrNull { it.radius }
            ?: return Scene(null, balls, table, tableFound, null, null, null, false)
        val cueOut = smooth(smoothCue, PointF(cueBall.x, cueBall.y), 0.55f) ?: PointF(cueBall.x, cueBall.y)
        smoothCue = cueOut

        val cuePositioned = Ball(cueOut.x, cueOut.y, cueBall.radius, true)

        // A ball whose radius disagrees with the rest is usually a highlight or a UI glyph.
        val coloured = balls.filter { !it.white }
        val radii = coloured.map { it.radius }.sorted()
        val median = if (radii.isEmpty()) cueBall.radius else radii[radii.size / 2]
        val consistent = coloured.filter {
            it.radius >= median * 0.55f && it.radius <= median * 1.8f
        }

        // The game draws its own guideline from the cue ball, so that line is the aim whenever it is
        // visible; the cue stick is the fallback, and the nearest ball the last resort.
        val lineAim = findAimLine(
            buffer,
            rowStride,
            pixelStride,
            limit,
            width,
            height,
            cueOut,
            cueBall.radius,
            table
        )
        val fromLine = lineAim != null
        val aimDirection = stableAim(lineAim ?: findStick(cueOut, components))
        var candidate: Ball? = null
        val secondary: Ball? = null
        val previousTarget = lastTarget
        if (aimDirection != null) {
            var bestScore = Float.MAX_VALUE
            for (ball in consistent) {
                val vx = ball.x - cueOut.x
                val vy = ball.y - cueOut.y
                val along = vx * aimDirection.x + vy * aimDirection.y
                if (along <= ball.radius * 1.2f) continue
                val across = abs(vx * aimDirection.y - vy * aimDirection.x)
                if (across > ball.radius * 2.5f) continue
                // A guideline stops at the first ball it crosses, so distance along it wins.
                var score = if (fromLine) along + across * 4f else across * 2f + along * 0.03f
                // Hold on to the target that was already on the line instead of flickering
                // between two balls at almost the same distance.
                if (previousTarget != null &&
                    hypot(ball.x - previousTarget.x, ball.y - previousTarget.y) < ball.radius * 2f
                ) {
                    score -= ball.radius * 6f
                }
                if (score < bestScore) {
                    bestScore = score
                    candidate = ball
                }
            }
            lastTarget = candidate?.let { PointF(it.x, it.y) }
        }

        val targetBall = candidate
        val hitSmoothed = if (targetBall != null) {
            val smoothed = smooth(smoothHit, PointF(targetBall.x, targetBall.y), 0.5f)
                ?: PointF(targetBall.x, targetBall.y)
            smoothHit = smoothed
            Ball(smoothed.x, smoothed.y, targetBall.radius, false)
        } else {
            smoothHit = null
            null
        }

        val all = ArrayList<Ball>()
        all.add(cuePositioned)
        all.addAll(consistent)
        return Scene(cuePositioned, all, table, tableFound, hitSmoothed, secondary, aimDirection, fromLine)
    }

    /**
     * Smooths the aim vector and refuses to jump.
     *
     * A single noisy frame must never swing the guide line across the table, so a new direction
     * has to be reported on three consecutive frames before it replaces the one on screen.
     */
    private fun stableAim(current: PointF?): PointF? {
        val previous = smoothAim
        if (current == null) {
            smoothAim = null
            pendingAim = null
            pendingAimFrames = 0
            return null
        }
        if (previous == null) {
            smoothAim = current
            return current
        }
        val dot = previous.x * current.x + previous.y * current.y
        if (dot >= 0.985f) {
            pendingAim = null
            pendingAimFrames = 0
            val blended = blend(previous, current, 0.35f) ?: current
            smoothAim = blended
            return blended
        }
        val pending = pendingAim
        if (pending != null && pending.x * current.x + pending.y * current.y >= 0.97f) {
            pendingAimFrames++
        } else {
            pendingAim = current
            pendingAimFrames = 1
        }
        if (pendingAimFrames >= 3) {
            pendingAim = null
            pendingAimFrames = 0
            smoothAim = current
            return current
        }
        return previous
    }

    private fun blend(previous: PointF, current: PointF, weight: Float): PointF? {
        val x = previous.x * (1f - weight) + current.x * weight
        val y = previous.y * (1f - weight) + current.y * weight
        val length = hypot(x, y)
        if (length < 0.0001f) return null
        return PointF(x / length, y / length)
    }

    /**
     * Finds the game's own white guideline and returns the direction it points in.
     *
     * The guideline is a thin, bright, unsaturated streak that starts at the cue ball, so the search
     * runs over a square around the cue ball, a RANSAC fit keeps only points on one straight line
     * through it, and the side holding more white pixels is taken as the forward direction.
     */
    private fun findAimLine(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        limit: Int,
        width: Int,
        height: Int,
        cue: PointF,
        cueRadius: Float,
        table: RectF
    ): PointF? {
        val reach = minOf(frameWidth, frameHeight) * 0.30f
        val minX = (cue.x - reach).toInt().coerceAtLeast(0)
        val maxX = (cue.x + reach).toInt().coerceAtMost(width - 1)
        val minY = (cue.y - reach).toInt().coerceAtLeast(0)
        val maxY = (cue.y + reach).toInt().coerceAtMost(height - 1)
        if (maxX - minX < 24 || maxY - minY < 24) return null

        val xs = FloatArray(3072)
        val ys = FloatArray(3072)
        var count = 0
        var y = minY
        while (y <= maxY) {
            val rowStart = y * rowStride
            var x = minX
            while (x <= maxX) {
                val offset = rowStart + x * pixelStride
                if (offset + 2 < limit && count < xs.size) {
                    val r = buffer.get(offset).toInt() and 0xFF
                    val g = buffer.get(offset + 1).toInt() and 0xFF
                    val b = buffer.get(offset + 2).toInt() and 0xFF
                    val low = minOf(r, g, b)
                    val high = maxOf(r, g, b)
                    if (low >= 228 && high - low <= 24) {
                        val dx = x - cue.x
                        val dy = y - cue.y
                        if (hypot(dx, dy) > cueRadius * 2f && table.contains(x.toFloat(), y.toFloat())) {
                            xs[count] = x.toFloat()
                            ys[count] = y.toFloat()
                            count++
                        }
                    }
                }
                x += 2
            }
            y += 2
        }
        if (count < 60) return null

        var bestScore = 0
        var bestX = 0f
        var bestY = 0f
        var seed = 7919
        var attempt = 0
        while (attempt < 80) {
            attempt++
            seed = seed * 1103515245 + 12345
            val i = (seed ushr 9) % count
            seed = seed * 1103515245 + 12345
            val j = (seed ushr 9) % count
            if (i == j) continue
            var vx = xs[j] - xs[i]
            var vy = ys[j] - ys[i]
            val span = hypot(vx, vy)
            if (span < cueRadius * 3f) continue
            vx /= span
            vy /= span
            val normalX = -vy
            val normalY = vx
            val cueDistance = abs((cue.x - xs[i]) * normalX + (cue.y - ys[i]) * normalY)
            if (cueDistance > cueRadius * 1.8f) continue
            var inliers = 0
            var forward = 0
            var backward = 0
            var lowest = Float.MAX_VALUE
            var highest = -Float.MAX_VALUE
            var k = 0
            while (k < count) {
                val px = xs[k] - xs[i]
                val py = ys[k] - ys[i]
                if (abs(px * normalX + py * normalY) <= 2.5f) {
                    inliers++
                    val along = (xs[k] - cue.x) * vx + (ys[k] - cue.y) * vy
                    if (along >= 0f) forward++ else backward++
                    if (along < lowest) lowest = along
                    if (along > highest) highest = along
                }
                k++
            }
            // A real guideline is long, one-sided and dense; scattered white glyphs are not.
            if (inliers < 55) continue
            if (inliers < count * 0.25f) continue
            if (forward < 12 || forward < backward * 1.4f) continue
            if (highest - lowest < cueRadius * 8f) continue
            if (inliers <= bestScore) continue
            bestScore = inliers
            bestX = vx
            bestY = vy
        }
        if (bestScore == 0) return null
        val length = hypot(bestX, bestY)
        if (length < 0.0001f) return null
        return PointF(bestX / length, bestY / length)
    }

    /**
     * Finds the cue stick: a long, thin blob lying next to the cue ball. The vector from the stick
     * towards the cue ball is the direction the shot is aimed, which is what makes the guide line
     * follow the player instead of pointing at whichever ball happens to be nearest.
     */
    private fun findStick(cue: PointF, components: List<Component>): PointF? {
        var stick: Component? = null
        var longest = 0f
        for (component in components) {
            val w = (component.maxX - component.minX + 1).toFloat()
            val h = (component.maxY - component.minY + 1).toFloat()
            val length = maxOf(w, h)
            val thin = minOf(w, h)
            if (thin <= 0f) continue
            if (length / thin < 3.2f) continue
            val lengthPx = length * step
            if (lengthPx < frameWidth * 0.10f) continue
            val dx = component.centreX(step) - cue.x
            val dy = component.centreY(step) - cue.y
            if (hypot(dx, dy) > lengthPx * 1.5f) continue
            if (lengthPx > longest) {
                longest = lengthPx
                stick = component
            }
        }
        val found = stick ?: return null
        val varianceX = found.meanSquareX - found.sumX * found.sumX
        val varianceY = found.meanSquareY - found.sumY * found.sumY
        val covariance = found.meanSquareXY - found.sumX * found.sumY
        val trace = varianceX + varianceY
        val determinant = varianceX * varianceY - covariance * covariance
        val lambda = trace / 2f + sqrt(maxOf(0f, trace * trace / 4f - determinant))
        var axisX = covariance
        var axisY = lambda - varianceX
        if (abs(axisX) < 0.0001f && abs(axisY) < 0.0001f) {
            axisX = 1f
            axisY = 0f
        }
        val axisLength = hypot(axisX, axisY)
        if (axisLength < 0.0001f) return null
        axisX /= axisLength
        axisY /= axisLength
        val toCueX = cue.x - found.centreX(step)
        val toCueY = cue.y - found.centreY(step)
        val sign = if (axisX * toCueX + axisY * toCueY >= 0f) 1f else -1f
        return PointF(axisX * sign, axisY * sign)
    }

    /** The most frequent coarse colour in the frame is the table cloth. */
    private fun dominantColour(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        limit: Int,
        width: Int,
        height: Int
    ): IntArray {
        val bins = IntArray(4096)
        var y = 0
        while (y < height) {
            val rowStart = y * rowStride
            var x = 0
            while (x < width) {
                val offset = rowStart + x * pixelStride
                if (offset + 2 < limit) {
                    val r = buffer.get(offset).toInt() and 0xFF
                    val g = buffer.get(offset + 1).toInt() and 0xFF
                    val b = buffer.get(offset + 2).toInt() and 0xFF
                    bins[((r shr 4) shl 8) or ((g shr 4) shl 4) or (b shr 4)]++
                }
                x += step
            }
            y += step
        }
        var bestIndex = 0
        var bestCount = -1
        for (index in bins.indices) {
            if (bins[index] > bestCount) {
                bestCount = bins[index]
                bestIndex = index
            }
        }
        val r = (((bestIndex shr 8) and 0xF) shl 4) + 8
        val g = (((bestIndex shr 4) and 0xF) shl 4) + 8
        val b = ((bestIndex and 0xF) shl 4) + 8
        return intArrayOf(r, g, b)
    }

    private fun labelCells(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        limit: Int,
        width: Int,
        height: Int,
        cloth: IntArray,
        tuning: Tuning
    ) {
        var y = 0
        while (y < height) {
            val rowStart = y * rowStride
            var x = 0
            while (x < width) {
                val offset = rowStart + x * pixelStride
                if (offset + 2 < limit) {
                    val r = buffer.get(offset).toInt() and 0xFF
                    val g = buffer.get(offset + 1).toInt() and 0xFF
                    val b = buffer.get(offset + 2).toInt() and 0xFF
                    val low = minOf(r, g, b)
                    val high = maxOf(r, g, b)
                    val spread = high - low
                    val far = abs(r - cloth[0]) + abs(g - cloth[1]) + abs(b - cloth[2])
                    val index = (y / step) * columns + (x / step)
                    if (index in cells.indices) {
                        if (low >= tuning.whiteFloor && spread <= tuning.whiteSpread) {
                            cells[index] = 1
                        } else if (high >= 70 && spread >= tuning.colourSpread && far >= tuning.clothDistance && cells[index] == 0) {
                            cells[index] = 2
                        }
                    }
                }
                x += step
            }
            y += step
        }
    }

    private fun collectComponents(): List<Component> {
        val found = ArrayList<Component>()
        var nextLabel = 0
        for (start in cells.indices) {
            if (cells[start] == 0 || labels[start] != 0) continue
            val kind = cells[start]
            nextLabel++
            var top = 0
            labels[start] = nextLabel
            stack[top++] = start

            fun push(index: Int) {
                if (index < 0 || index >= cells.size) return
                if (cells[index] != kind || labels[index] != 0) return
                labels[index] = nextLabel
                stack[top++] = index
            }

            var count = 0
            var sumX = 0
            var sumY = 0
            var squareX = 0.0
            var squareXY = 0.0
            var squareY = 0.0
            var minX = columns
            var maxX = -1
            var minY = rows
            var maxY = -1

            while (top > 0) {
                val current = stack[--top]
                val cx = current % columns
                val cy = current / columns
                count++
                sumX += cx
                sumY += cy
                squareX += cx.toDouble() * cx
                squareXY += cx.toDouble() * cy
                squareY += cy.toDouble() * cy
                if (cx < minX) minX = cx
                if (cx > maxX) maxX = cx
                if (cy < minY) minY = cy
                if (cy > maxY) maxY = cy

                push(current - 1)
                push(current + 1)
                push(current - columns)
                push(current + columns)
            }

            if (count > 0) {
                found.add(
                    Component(
                        kind,
                        count,
                        sumX.toFloat() / count,
                        sumY.toFloat() / count,
                        (squareX / count).toFloat(),
                        (squareXY / count).toFloat(),
                        (squareY / count).toFloat(),
                        minX,
                        minY,
                        maxX,
                        maxY
                    )
                )
            }
        }
        return found
    }

    /** Mean colour of a small patch, used to tell the white cue ball from the coloured balls. */
    private fun samplesWhite(
        buffer: ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        limit: Int,
        cx: Int,
        cy: Int
    ): Boolean {
        var red = 0
        var green = 0
        var blue = 0
        var count = 0
        var dy = -3
        while (dy <= 3) {
            var dx = -3
            while (dx <= 3) {
                val offset = (cy + dy) * rowStride + (cx + dx) * pixelStride
                if (offset + 2 < limit) {
                    red += buffer.get(offset).toInt() and 0xFF
                    green += buffer.get(offset + 1).toInt() and 0xFF
                    blue += buffer.get(offset + 2).toInt() and 0xFF
                    count++
                }
                dx++
            }
            dy++
        }
        if (count == 0) return false
        val low = minOf(red / count, green / count, blue / count)
        val high = maxOf(red / count, green / count, blue / count)
        return low >= 150 && high - low <= 45
    }

    private fun smooth(previous: PointF?, current: PointF, keep: Float): PointF? {
        if (previous == null) return null
        if (hypot(previous.x - current.x, previous.y - current.y) > 400f) return null
        return PointF(
            previous.x * keep + current.x * (1f - keep),
            previous.y * keep + current.y * (1f - keep)
        )
    }
}
