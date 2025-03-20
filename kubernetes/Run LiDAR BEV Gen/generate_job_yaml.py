import os
import yaml
def generate_yaml(filename):
  job_template = f"""
apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}{number}
  namespace: ecewcsng
spec:
  # replicas: 1
  # selector:
  #   matchLabels:
  #     app: {job_name}{number}
  template:
    metadata:
      labels:
        app: {job_name}{number}
    spec:
      hostname: {job_name}{number}
#      hostNetwork: true
      restartPolicy: Never
      containers:
      - name: {job_name}{number}
        image: gitlab-registry.nrp-nautilus.io/wcsng/wcsng-radarimaging:58a9405d
        command:
        - /bin/sh
        - -c
        - "cd /radar-imaging-dataset/{dir_name}/LiDAR_Scripts && bash job{number}.sh"
#        securityContext:
#          privileged: true
        env:
        - name: TZ
          value: "UTC"
        # Keep to default unless you know what you are doing with VirtualGL, `VGL_DISPLAY` should be set to either `egl[n]`, or `/dev/dri/card[n]` only when the device was passed to the container
#        - name: VGL_DISPLAY
#          value: "egl"
        - name: SIZEW
          value: "1920"
        - name: SIZEH
          value: "1200"
        - name: REFRESH
          value: "60"
        - name: DPI
          value: "96"
        - name: CDEPTH
          value: "24"
        # Choose either `value:` or `secretKeyRef:` but not both at the same time
        - name: PASSWD
#          value: "mypasswd"
          valueFrom:
            secretKeyRef:
              name: my-pass
              key: my-pass
        # Uncomment this to enable noVNC, disabing selkies-gstreamer and ignoring all its parameters except `BASIC_AUTH_PASSWORD`, which will be used for authentication with noVNC, `BASIC_AUTH_PASSWORD` defaults to `PASSWD` if not provided
        - name: NOVNC_ENABLE
          value: "true"
        # Additional view-only password only applicable to the noVNC interface, choose either `value:` or `secretKeyRef:` but not both at the same time
#        - name: NOVNC_VIEWPASS
#          value: "mypasswd"
#          valueFrom:
#            secretKeyRef:
#              name: my-pass
#              key: my-pass
        ###
        # selkies-gstreamer parameters, for additional configurations see lines that start with "parser.add_argument" in https://github.com/selkies-project/selkies-gstreamer/blob/master/src/selkies_gstreamer/__main__.py
        ###
        # Change `WEBRTC_ENCODER` to `x264enc` if you are using software fallback without allocated GPUs or your GPU doesn't support `H.264 (AVCHD)` under the `NVENC - Encoding` section in https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
        - name: WEBRTC_ENCODER
#          value: "nvh264enc"
          value: "x264enc"
        - name: WEBRTC_ENABLE_RESIZE
          value: "false"
        - name: ENABLE_AUDIO
          value: "true"
        - name: ENABLE_BASIC_AUTH
          value: "true"
        # Defaults to `PASSWD` if unspecified, choose either `value:` or `secretKeyRef:` but not both at the same time
#        - name: BASIC_AUTH_PASSWORD
#          value: "mypasswd"
#          valueFrom:
#            secretKeyRef:
#              name: my-pass
#              key: my-pass
        ###
        # Uncomment below to use a TURN server for improved network compatibility
        ###
        - name: TURN_HOST
          value: "turn.nrp-nautilus.io"
        - name: TURN_PORT
          value: "3478"
        # Provide only `TURN_SHARED_SECRET` for time-limited shared secret authentication or both `TURN_USERNAME` and `TURN_PASSWORD` for legacy long-term authentication, but do not provide both authentication methods at the same time
        - name: TURN_SHARED_SECRET
          valueFrom:
            secretKeyRef:
              name: my-pass
              key: turn-secret
#        - name: TURN_USERNAME
#          value: "username"
        # Choose either `value:` or `secretKeyRef:` but not both at the same time
#        - name: TURN_PASSWORD
#          value: "mypasswd"
#          valueFrom:
#            secretKeyRef:
#              name: turn-password
#              key: turn-password
        # Change to `tcp` if the UDP protocol is throttled or blocked in your client network, or when the TURN server does not support UDP
        - name: TURN_PROTOCOL
          value: "udp"
        # You need a valid hostname and a certificate from authorities such as ZeroSSL (Let's Encrypt may have issues) to enable this
        - name: TURN_TLS
          value: "false"
        stdin: true
        tty: true
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        resources:
          limits:
            memory: 2Gi
            cpu: "4"
            # nvidia.com/gpu: 1
          requests:
            memory: 1Gi
            cpu: "2"
            # nvidia.com/gpu: 1
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
        - mountPath: /radar-imaging-dataset
          name: radar-imaging-dataset
        - mountPath: /cache
          name: egl-cache-vol
      dnsPolicy: None
      dnsConfig:
        nameservers:
        - 8.8.8.8
        - 8.8.4.4
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
      - name: egl-cache-vol
        emptyDir: {{}}
#        persistentVolumeClaim:
#          claimName: egl-cache-vol
      - name: radar-imaging-dataset
        persistentVolumeClaim:
          claimName: radar-imaging-dataset
      # affinity:
      #   nodeAffinity:
      #     requiredDuringSchedulingIgnoredDuringExecution:
      #       nodeSelectorTerms:
      #       - matchExpressions:
              # - key: nvidia.com/gpu.product
              #   operator: In
              #   values:
              #   - NVIDIA-A10
      tolerations:
      - effect: NoSchedule
        key: nautilus.io/nrp-testing
        operator: Exists
"""
  return job_template
  
if __name__ == "__main__":
    MAX_JOBS = 162

    job_name = "lidar-bev-gen"
    dir_name = "P2SIF/data-collection/Scripts"
    
    for number in range(MAX_JOBS):
        job_content = generate_yaml(number)
        yaml_content = yaml.safe_load(job_content)
        os.makedirs(f"./{job_name}", exist_ok = True)
        
        with open(f"./{job_name}/job{number}.yaml", "w") as yaml_file:
            yaml.dump(yaml_content, yaml_file, default_flow_style=False)