"""
UDP Network Module
Handles UDP socket communication with Godot clients.
Manages client registration and frame broadcasting.
"""

import socket
import threading


class UDPServer:
    """
    UDP server that broadcasts video frames to registered Godot clients.
    Handles client registration/unregistration and packet transmission.
    """
    
    def __init__(self, host='127.0.0.1', port=9999):
        """
        Initializes UDP server.
        
        Args:
            host: Server IP address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.socket = None
        self.clients = set()  # Set of (ip, port) tuples
        self.lock = threading.Lock()
        self.is_running = False
        
        # Maximum UDP packet size (64KB minus headers)
        self.max_packet_size = 65507
        
    def start(self):
        """
        Starts the UDP server and begins listening for client messages.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(0.1)  # Non-blocking with timeout
            self.is_running = True
            
            print(f"[INFO] UDP Server started on {self.host}:{self.port}")
            
            # Start listener thread for client commands
            self.listener_thread = threading.Thread(target=self._listen_for_clients, daemon=True)
            self.listener_thread.start()
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to start UDP server: {e}")
            return False
    
    def _listen_for_clients(self):
        """
        Listens for REGISTER and UNREGISTER commands from clients.
        Runs in a separate thread.
        """
        while self.is_running:
            try:
                data, client_addr = self.socket.recvfrom(1024)
                message = data.decode('utf-8').strip()
                
                if message == "REGISTER":
                    self._register_client(client_addr)
                elif message == "UNREGISTER":
                    self._unregister_client(client_addr)
                    
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    print(f"[WARNING] Listener error: {e}")
    
    def _register_client(self, client_addr):
        """
        Registers a new client for frame streaming.
        
        Args:
            client_addr: tuple (ip, port)
        """
        with self.lock:
            if client_addr not in self.clients:
                self.clients.add(client_addr)
                print(f"[INFO] Client registered: {client_addr}")
    
    def _unregister_client(self, client_addr):
        """
        Unregisters a client from frame streaming.
        
        Args:
            client_addr: tuple (ip, port)
        """
        with self.lock:
            if client_addr in self.clients:
                self.clients.remove(client_addr)
                print(f"[INFO] Client unregistered: {client_addr}")
    
    def broadcast_packet(self, packet):
        """
        Sends a packet to all registered clients.
        
        Args:
            packet: bytes to send
            
        Returns:
            int: Number of clients that received the packet
        """
        if packet is None or len(packet) == 0:
            return 0
        
        if len(packet) > self.max_packet_size:
            print(f"[WARNING] Packet too large: {len(packet)} bytes (max: {self.max_packet_size})")
            return 0
        
        sent_count = 0
        
        with self.lock:
            clients_to_remove = []
            
            for client_addr in self.clients:
                try:
                    self.socket.sendto(packet, client_addr)
                    sent_count += 1
                except Exception as e:
                    print(f"[WARNING] Failed to send to {client_addr}: {e}")
                    clients_to_remove.append(client_addr)
            
            # Remove clients that failed to receive
            for client_addr in clients_to_remove:
                self.clients.remove(client_addr)
                print(f"[INFO] Client removed due to send failure: {client_addr}")
        
        return sent_count
    
    def get_client_count(self):
        """
        Returns the number of registered clients.
        
        Returns:
            int: Client count
        """
        with self.lock:
            return len(self.clients)
    
    def stop(self):
        """
        Stops the UDP server and closes the socket.
        """
        self.is_running = False
        
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        
        with self.lock:
            self.clients.clear()
        
        print("[INFO] UDP Server stopped")
    
    def __del__(self):
        """Ensures proper cleanup when object is destroyed."""
        self.stop()
