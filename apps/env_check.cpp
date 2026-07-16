// Environment smoke test: links and exercises every core C++ dependency.
// CI runs this as the acceptance gate for docker/Dockerfile.core.
// Exit code 0 = environment healthy.

#include <CLI/CLI.hpp>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <nlohmann/json.hpp>
#include <open3d/Open3D.h>
#include <open3d/core/Tensor.h>
#include <open3d/t/geometry/PointCloud.h>
#include <yaml-cpp/yaml.h>

#include <cstdio>

namespace {

// Trivial Ceres problem: minimize (x - 4)^2 → x = 4.
struct QuadCost {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = x[0] - T(4.0);
        return true;
    }
};

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"blockpose environment check"};
    CLI11_PARSE(app, argc, argv);

    // Eigen: SE(3) round trip.
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.rotate(Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()));
    T.pretranslate(Eigen::Vector3d(0.9, 0.6, 0.6));
    const Eigen::Isometry3d I = T * T.inverse();
    if (!I.isApprox(Eigen::Isometry3d::Identity(), 1e-12)) {
        std::fprintf(stderr, "FAIL: Eigen SE(3) round trip\n");
        return 1;
    }

    // Open3D tensor API (float32 — the committed core convention, plan §2.3).
    open3d::core::Tensor pts =
        open3d::core::Tensor::Init<float>({{0.f, 0.f, 0.f},
                                           {0.9f, 0.f, 0.f},
                                           {0.f, 0.6f, 0.f},
                                           {0.f, 0.f, 0.6f}});
    open3d::t::geometry::PointCloud cloud(pts);
    if (cloud.GetPointPositions().GetDtype() != open3d::core::Float32) {
        std::fprintf(stderr, "FAIL: Open3D tensor dtype\n");
        return 1;
    }
    const auto down = cloud.VoxelDownSample(0.05);
    if (down.GetPointPositions().GetLength() != 4) {
        std::fprintf(stderr, "FAIL: Open3D voxel downsample\n");
        return 1;
    }

    // Ceres: one-variable solve.
    double x = 0.0;
    ceres::Problem problem;
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<QuadCost, 1, 1>(new QuadCost), nullptr,
        &x);
    ceres::Solver::Options options;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (std::abs(x - 4.0) > 1e-6) {
        std::fprintf(stderr, "FAIL: Ceres solve (x=%f)\n", x);
        return 1;
    }

    // yaml-cpp + nlohmann_json round trips.
    const YAML::Node y = YAML::Load("block: {dims: [0.9, 0.6, 0.6]}");
    const nlohmann::json j = {{"schema_version", 1}};
    if (y["block"]["dims"][0].as<double>() != 0.9 || j["schema_version"] != 1) {
        std::fprintf(stderr, "FAIL: yaml/json round trip\n");
        return 1;
    }

    std::printf("env_check OK\n");
    std::printf("  Eigen   %d.%d.%d\n", EIGEN_WORLD_VERSION,
                EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
    std::printf("  Open3D  %s\n", OPEN3D_VERSION);
    std::printf("  Ceres   %s\n", CERES_VERSION_STRING);
    return 0;
}
