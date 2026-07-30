#include <blockpose/core/refine.hpp>

#include <gtest/gtest.h>

namespace blockpose::core { namespace {

Points surface(const CuboidPose & pose) {Points p; Point h(pose.dims[0]/2.,pose.dims[1]/2.,pose.dims[2]/2.); auto add=[&](Point local){p.push_back(pose.position+pose.rotation*local);}; for(int x=-4;x<=4;++x)for(int y=-3;y<=3;++y)add(Point(h.x()*x/4.,h.y()*y/3.,h.z())); for(int y=-3;y<=3;++y)for(int z=-3;z<=3;++z)add(Point(h.x(),h.y()*y/3.,h.z()*z/3.)); return p;}
TEST(Refine, SdfAndExpMatchPrototype) {Point h(.45,.3,.3); EXPECT_NEAR(box_sdf(Point(.45,0,0),h),0,1e-12); EXPECT_NEAR(box_sdf(Point(0,0,0),h),-.3,1e-12); EXPECT_NEAR(box_sdf(Point(.55,0,0),h),.1,1e-12); EXPECT_NEAR(rotation_angle_deg(exp_so3(Point(0,0,.2))),11.4591559,1e-6);}
TEST(Refine, AssignmentUsesNearestSurfaceAndBand) {CuboidPose a,b;a.position=Point(-1,0,0);b.position=Point(1,0,0);auto owned=assign_points({a,b},{Point(-.55,0,0),Point(1.45,0,0),Point(0,4,0)},.1);EXPECT_EQ(owned[0].size(),1U);EXPECT_EQ(owned[1].size(),1U);}
TEST(Refine, RobustFitCorrectsSmallPoseError) {CuboidPose truth;truth.position=Point(.2,-.1,.3);truth.rotation=exp_so3(Point(0,0,.22));truth.source="synthetic";const auto points=surface(truth);CuboidPose initial=truth;initial.position+=Point(.045,-.035,.02);initial.rotation*=exp_so3(Point(0,0,-.1));const auto result=refine_pose(initial,points);ASSERT_TRUE(result.refined);EXPECT_LT((result.pose.position-truth.position).norm(),.006);EXPECT_LT(rotation_angle_deg(truth.rotation.transpose()*result.pose.rotation),.8);EXPECT_EQ(result.pose.source,"synthetic+sdf");}

TEST(Refine, AnalyticJacobianMatchesCentralDifferences)
{
  CuboidPose base;
  base.position = Point(.2, -.1, .35);
  base.rotation = exp_so3(Point(.05, -.03, .22));
  const Point half(base.dims[0] / 2., base.dims[1] / 2., base.dims[2] / 2.);
  Points probes;
  for (int u = -2; u <= 2; ++u) {
    for (int v = -2; v <= 2; ++v) {
      probes.push_back(base.position + base.rotation * Point(half.x() * u / 3., half.y() * v / 3., half.z() * 1.15));
      probes.push_back(base.position + base.rotation * Point(half.x() * u / 3., half.y() * v / 3., half.z() * .55));
    }
  }
  for (const auto & offset : {Point(0., 0., 0.), Point(.02, -.01, .015), Point(-.03, .012, -.02)}) {
    for (const auto & rotation_vector : {Point(0., 0., 0.), Point(.03, -.02, .12), Point(-.09, .05, -.18)}) {
      const Eigen::Matrix3d r = base.rotation * exp_so3(rotation_vector);
      const Eigen::Matrix3d jr = right_jacobian_so3(rotation_vector);
      const auto residual_at = [&](const Point & rv, const Point & t, const Point & p) {
        return box_sdf((base.rotation * exp_so3(rv)).transpose() * (p - (base.position + t)), half);
      };
      for (const auto & point : probes) {
        const Point q = r.transpose() * (point - (base.position + offset));
        Point gradient;
        box_sdf_gradient(q, half, &gradient);
        Eigen::Matrix<double, 1, 6> analytic;
        analytic.block<1, 3>(0, 0) = (gradient.transpose() * skew(q)) * jr;
        analytic.block<1, 3>(0, 3) = -(gradient.transpose() * r.transpose());
        constexpr double h = 1.0e-6;
        for (int column = 0; column < 6; ++column) {
          Point rv_plus = rotation_vector, rv_minus = rotation_vector, t_plus = offset, t_minus = offset;
          if (column < 3) {rv_plus[column] += h; rv_minus[column] -= h;} else {t_plus[column - 3] += h; t_minus[column - 3] -= h;}
          const double numeric = (residual_at(rv_plus, t_plus, point) - residual_at(rv_minus, t_minus, point)) / (2. * h);
          EXPECT_NEAR(analytic[column], numeric, 2.0e-6)
            << "column=" << column << " rv=" << rotation_vector.transpose() << " t=" << offset.transpose();
        }
      }
    }
  }
}

TEST(Refine, FailuresKeepInitialPose) {CuboidPose pose;EXPECT_FALSE(refine_pose(pose,{Point::Zero()}).refined);const auto points=surface(pose);CuboidPose far=pose;far.position.x()=.5;const auto guarded=refine_pose(far,points);EXPECT_FALSE(guarded.refined);EXPECT_NEAR(guarded.pose.position.x(),.5,1e-12);}

}}  // namespace
