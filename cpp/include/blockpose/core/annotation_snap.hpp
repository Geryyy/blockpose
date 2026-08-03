#pragma once

// Native local cuboid fitting used by the annotation UI.  Keeping this here
// prevents annotation convenience code from becoming a second geometry core.
#include <blockpose/core/refine.hpp>

#include <array>
#include <optional>

namespace blockpose::core
{
struct AnnotationSnapParameters {
  double crop_radius_m{1.05}, voxel_m{0.018}, plane_tolerance_m{0.035}, surface_tolerance_m{0.045};
  std::size_t min_points{45}, min_inliers{30}; double min_score{0.30}, ambiguity_score_ratio{0.96}, ambiguity_distance_m{0.08}; bool reject_ambiguous{true};
};
struct AnnotationSnapResult {
  bool success{false}; CuboidPose pose{}; double score{0.0}; std::size_t inlier_count{0}; std::string reason;
};
struct AnnotationPlane {Point normal; double offset; Points points;};

inline std::optional<std::string> validate_annotation_snap_parameters(const AnnotationSnapParameters & p)
{
  const auto positive = [](double v) {return std::isfinite(v) && v > 0.0;};
  if (!positive(p.crop_radius_m) || !positive(p.voxel_m) || !positive(p.plane_tolerance_m) || !positive(p.surface_tolerance_m) || !positive(p.ambiguity_distance_m)) return "snap parameters must be finite and positive";
  if (p.min_points < 3U || p.min_inliers < 3U) return "min_points and min_inliers must be at least 3";
  if (!std::isfinite(p.min_score) || p.min_score < 0.0 || p.min_score > 1.0) return "min_score must lie in [0, 1]";
  if (!std::isfinite(p.ambiguity_score_ratio) || p.ambiguity_score_ratio <= 0.0 || p.ambiguity_score_ratio > 1.0) return "ambiguity_score_ratio must lie in (0, 1]";
  if (p.voxel_m >= p.crop_radius_m) return "voxel_m must be smaller than crop_radius_m";
  if (std::max(p.plane_tolerance_m, p.surface_tolerance_m) >= p.crop_radius_m) return "fit tolerances must be smaller than crop_radius_m";
  return std::nullopt;
}
inline Points annotation_voxels(const Points & input, double size)
{
  std::map<std::array<long long, 3>, std::pair<Point, std::size_t>> bins;
  for (const auto & q : canonical_order(input)) {
    const std::array<long long, 3> key{{
      static_cast<long long>(std::floor(q.x() / size)),
      static_cast<long long>(std::floor(q.y() / size)),
      static_cast<long long>(std::floor(q.z() / size)),
    }};
    // Eigen's default vector constructor leaves coefficients uninitialized.
    // Initialize the accumulator explicitly instead of relying on map::operator[].
    auto [it, inserted] = bins.try_emplace(key, Point::Zero(), 0U);
    it->second.first += q;
    ++it->second.second;
  }
  Points out; out.reserve(bins.size()); for (const auto & b : bins) out.push_back(b.second.first / static_cast<double>(b.second.second)); return out;
}
inline std::vector<Points> annotation_components(const Points & p, double radius)
{
  using Cell = std::array<long long, 3>;
  std::vector<std::size_t> parent(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) {parent[i] = i;}
  const auto root = [&parent](std::size_t index) {
    while (parent[index] != index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  };
  const auto cell_for = [radius](const Point & point) {
    return Cell{{
      static_cast<long long>(std::floor(point.x() / radius)),
      static_cast<long long>(std::floor(point.y() / radius)),
      static_cast<long long>(std::floor(point.z() / radius)),
    }};
  };
  std::map<Cell, std::vector<std::size_t>> cells;
  for (std::size_t index = 0; index < p.size(); ++index) {
    const Cell cell = cell_for(p[index]);
    for (long long dx = -1; dx <= 1; ++dx) {
      for (long long dy = -1; dy <= 1; ++dy) {
        for (long long dz = -1; dz <= 1; ++dz) {
          const Cell neighbor{{cell[0] + dx, cell[1] + dy, cell[2] + dz}};
          const auto found = cells.find(neighbor);
          if (found == cells.end()) {continue;}
          for (const std::size_t other : found->second) {
            if ((p[index] - p[other]).norm() > radius) {continue;}
            const std::size_t first = root(index), second = root(other);
            if (first != second) {parent[std::max(first, second)] = std::min(first, second);}
          }
        }
      }
    }
    cells[cell].push_back(index);
  }
  std::map<std::size_t, Points> groups;
  for (std::size_t index = 0; index < p.size(); ++index) {groups[root(index)].push_back(p[index]);}
  std::vector<Points> output;
  output.reserve(groups.size());
  for (auto & group : groups) {output.push_back(std::move(group.second));}
  return output;
}
inline std::vector<AnnotationPlane> annotation_planes(Points remaining, double tolerance)
{
  // segment_plane returns indices into its canonicalized input; retain that
  // ordering between extraction rounds before consuming those indices.
  remaining = canonical_order(std::move(remaining));
  std::vector<AnnotationPlane> out;
  for (std::size_t round=0; round<10U && remaining.size()>=18U; ++round) {
    PlaneFitResult fit; try {fit=segment_plane(remaining, {tolerance, 500U, round, .9999, 0U});} catch (...) {break;}
    if(fit.inlier_indices.size()<18U) break; Point n=fit.plane.normal; double d=fit.plane.offset; Eigen::Index pivot; n.cwiseAbs().maxCoeff(&pivot); if(n[pivot]<0){n=-n;d=-d;}
    Points consensus, next; std::vector<bool> hit(remaining.size()); for(auto i:fit.inlier_indices){consensus.push_back(remaining[i]);hit[i]=true;} for(std::size_t i=0;i<remaining.size();++i)if(!hit[i])next.push_back(remaining[i]);
    for(auto & c:annotation_components(consensus,std::max(.065,tolerance*2.5))) if(c.size()>=16U) out.push_back({n,d,std::move(c)}); remaining=std::move(next);
  } return out;
}
inline double annotation_span(const Points & p, const Point & axis, double lo, double hi) {std::vector<double> v; for(const auto&q:p)v.push_back(q.dot(axis)); return percentile_linear(v,hi)-percentile_linear(v,lo);}
inline bool annotation_overlap(const AnnotationPlane & top,const AnnotationPlane & side,const Point & tangent,double top_z)
{
  if(std::max_element(side.points.begin(),side.points.end(),[](const Point&a,const Point&b){return a.z()<b.z();})->z() < top_z-.16 || annotation_span(side.points,Point::UnitZ(),0,100)<.16) return false;
  const double a0=percentile_linear([&] {std::vector<double>v;for(auto&q:top.points)v.push_back(q.dot(tangent));return v;}(),3),a1=percentile_linear([&] {std::vector<double>v;for(auto&q:top.points)v.push_back(q.dot(tangent));return v;}(),97),b0=percentile_linear([&] {std::vector<double>v;for(auto&q:side.points)v.push_back(q.dot(tangent));return v;}(),3),b1=percentile_linear([&] {std::vector<double>v;for(auto&q:side.points)v.push_back(q.dot(tangent));return v;}(),97);
  Point n=side.normal; n.z()=0; n.normalize(); double coord=percentile_linear([&]{std::vector<double>v;for(auto&q:side.points)v.push_back(q.dot(n));return v;}(),50); std::size_t t=0,s=0; for(auto&q:top.points)if(std::abs(q.dot(n)-coord)<=.065)++t;for(auto&q:side.points)if(std::abs(q.z()-top_z)<=.16)++s; return std::min(a1,b1)>=std::max(a0,b0)-.05 && t>=3U && s>=6U;
}
inline std::tuple<double,std::size_t,Points> annotation_score(const Points & top,const Points & side,const CuboidPose & pose,int side_axis,double side_sign,double tol)
{
  const Point half(pose.dims[0]/2,pose.dims[1]/2,pose.dims[2]/2); Points assigned; std::vector<double> residuals; std::size_t tc=0,sc=0; const int other=1-side_axis;
  for(const auto&q:top){Point l=pose.rotation.transpose()*(q-pose.position);double r=std::abs(l.z()-half.z());if(std::abs(l.x())<=half.x()+tol&&std::abs(l.y())<=half.y()+tol&&r<=tol){++tc;assigned.push_back(q);residuals.push_back(r);}}
  for(const auto&q:side){Point l=pose.rotation.transpose()*(q-pose.position);double r=std::abs(l[side_axis]-side_sign*half[side_axis]);if(std::abs(l[other])<=half[other]+tol&&std::abs(l.z())<=half.z()+tol&&r<=tol){++sc;assigned.push_back(q);residuals.push_back(r);}}
  if(tc<16||sc<16)return {0.0,tc+sc,{}}; double quality=0;for(double r:residuals)quality+=std::max(0.0,1-r/tol);quality/=residuals.size();const double balance=2.*std::min(tc,sc)/(tc+sc),support=std::min((tc+sc)/180.,1.),coverage=double(tc)/top.size();return {quality*(.4+.35*support+.25*balance)*(.35+.65*coverage),assigned.size(),assigned};
}
inline AnnotationSnapResult annotation_snap(const Points & input,const Point & seed,const AnnotationSnapParameters & p={})
{
  if(auto error=validate_annotation_snap_parameters(p))return {false,{},0,0,*error}; Points local;for(auto&q:input)if(q.allFinite()&&(q-seed).norm()<=p.crop_radius_m)local.push_back(q);if(local.size()<p.min_points)return {false,{},0,local.size(),"only " + std::to_string(local.size()) + " local points; need " + std::to_string(p.min_points)}; local=annotation_voxels(local,p.voxel_m);if(local.size()<p.min_points)return {false,{},0,local.size(),"only " + std::to_string(local.size()) + " local voxels; need " + std::to_string(p.min_points)};
  const auto planes=annotation_planes(local,p.plane_tolerance_m); std::vector<AnnotationPlane> tops,sides;for(auto&x:planes){if(std::abs(x.normal.z())>=.94)tops.push_back(x);if(std::abs(x.normal.z())<=.18)sides.push_back(x);}if(tops.empty())return {false,{},0,0,"no finite horizontal top patch near the click"};if(sides.empty())return {false,{},0,0,"no compatible vertical front/side patch near the click"};
  struct Candidate{double score;std::size_t count;CuboidPose pose;};std::vector<Candidate> candidates; const Point z=Point::UnitZ(); const std::array<std::array<double,3>,2> dimensions{{{{.9,.6,.6}},{{.6,.6,.9}}}};
  for(const auto&top:tops){double top_z=percentile_linear([&]{std::vector<double>v;for(auto&q:top.points)v.push_back(q.z());return v;}(),50);if(top_z<seed.z()-.1)continue;for(const auto&side:sides){Point n=side.normal;n.z()=0;if(n.norm()<1e-12)continue;n.normalize();Point tangent=z.cross(n);const double side_height=annotation_span(side.points,z,2,98);if(annotation_span(side.points,tangent,2,98)>1.2||side_height>1.2||!annotation_overlap(top,side,tangent,top_z))continue;double coordinate=percentile_linear([&]{std::vector<double>v;for(auto&q:side.points)v.push_back(q.dot(n));return v;}(),50);for(const auto&dims:dimensions)for(int axis=0;axis<2;++axis)for(double sign:{-1.,1.}){if(side_height>dims[2]+.12)continue;Point ax=axis==0?n:tangent,ay=axis==0?tangent:-n; Eigen::Matrix3d r;r.col(0)=ax;r.col(1)=ay;r.col(2)=z;Point ext(annotation_span(top.points,ax,3,97),annotation_span(top.points,ay,3,97),0);if(ext.x()>dims[0]+.22||ext.y()>dims[1]+.22||ext.x()<.16||ext.y()<.16)continue;int other=1-axis;Point face=r.col(axis),other_axis=r.col(other),center=coordinate*n-sign*dims[axis]/2.*face;std::vector<double> vals;for(auto&q:top.points)vals.push_back(q.dot(other_axis));std::array<double,5> centers{{(percentile_linear(vals,4)+percentile_linear(vals,96))/2,percentile_linear(vals,4)+dims[other]/2,percentile_linear(vals,96)-dims[other]/2,percentile_linear(vals,50),seed.dot(other_axis)}};for(double value:centers){CuboidPose pose;pose.position=center+(value-center.dot(other_axis))*other_axis;pose.position.z()=top_z-dims[2]/2;pose.rotation=r;pose.dims=dims;Point sl=r.transpose()*(seed-pose.position);if((sl.cwiseAbs()-Point(dims[0]/2+.1,dims[1]/2+.1,dims[2]/2+.1)).maxCoeff()>0)continue;auto [score,count,assigned]=annotation_score(top.points,side.points,pose,axis,sign,p.surface_tolerance_m);if(count<p.min_inliers)continue;pose=refine_pose(pose,assigned,{.05,.06,6.,p.min_inliers,200U}).pose;std::tie(score,count,std::ignore)=annotation_score(top.points,side.points,pose,axis,sign,p.surface_tolerance_m);candidates.push_back({score,count,pose});}}}}
  if(candidates.empty())return {false,{},0,0,"no top/side intersection supports a canonical block"};std::sort(candidates.begin(),candidates.end(),[](const auto&a,const auto&b){if(a.score!=b.score)return a.score>b.score;if(a.count!=b.count)return a.count>b.count;return canonical_less(a.pose.position,b.pose.position);});const auto&best=candidates.front();if(best.count<p.min_inliers)return {false,{},0,best.count,"insufficient cuboid surface support ("+std::to_string(best.count)+" inliers)"};if(best.score<p.min_score)return {false,{},0,best.count,"cuboid fit score below minimum"};if(p.reject_ambiguous)for(auto it=std::next(candidates.begin());it!=candidates.end()&&it->score>=best.score*p.ambiguity_score_ratio;++it)if((it->pose.position-best.pose.position).norm()>=p.ambiguity_distance_m)return {false,{},0,0,"ambiguous fit: near-equal candidates are apart"};
  // Convert upright model axes to the canonical laid 0.9x0.6x0.6 convention.
  CuboidPose out=best.pose;if(best.pose.dims[2]>.7){Eigen::Matrix3d a; a<<0,0,1, 0,1,0, -1,0,0;out.rotation*=a;}out.dims={{.9,.6,.6}};return {true,out,best.score,best.count,{}};
}
}  // namespace blockpose::core
