import grpc
from concurrent import futures
from proto import vision_pb2
from proto import vision_pb2_grpc


class HapticFeedbackServicer(vision_pb2_grpc.HapticFeedbackServicer):
    def SendDetection(self, request, context):
        print(len(request.boxes))
        return vision_pb2.DetectionResponse(ok=True, message="received")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    vision_pb2_grpc.add_HapticFeedbackServicer_to_server(HapticFeedbackServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
