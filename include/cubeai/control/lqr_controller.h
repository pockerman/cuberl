//
// Created by alex on 10/11/25.
//

#ifndef LQR_CONTROLLER_H
#define LQR_CONTROLLER_H

#include "cubeai/base/cubeai_types.h"

#include <vector>

namespace cuberl
{
    namespace control
    {
        template<uint_t StateSize, uint_t InputSize, typename T=real_t>
        struct LQRControlSysDescription
        {
            SquareMat<T, StateSize> A;
            SquareMat<T, StateSize> Q;
            SquareMat<T, StateSize> R;
            Mat<T, StateSize, InputSize> InputMat;
        };

        template<typename StateType, uint_t StateSize, uint_t InputSize, typename T=real_t>
        class LQRController
        {
        public:

            typedef StateType state_type;
            typedef std::vector<state_type> ctrl_signal_type;

            ///  Constructor
            /// @param sys_description
            LQRController(const LQRControlSysDescription<StateSize, InputSize, T>& sys_description);

            ctrl_signal_type compute_ctrl_signal(const StateType& init, const StateType& target, real_t dt);

            void set_time_limit(real_t limit) { max_time_ = limit; }
            void set_final_position_tolerance(real_t tol) { goal_dist_ = tol; }
            void set_iterations_limit(uint_t limit) { max_iter_ = limit; }
            void set_tolerance(real_t tol) { eps_ = tol; }

        private:

            LQRControlSysDescription<StateSize, InputSize, T> sys_description_;
            Mat<T, StateSize, InputSize> k_;
            uint_t max_iter_{10};
            real_t max_time_{100.0};
            real_t goal_dist_{0.1};
            real_t eps_{0.01};

            void update_gain_mat();
            SquareMat<T, StateSize> solve_lqr();

        };

        template<typename StateType, uint_t StateSize, uint_t InputSize, typename T>
        LQRController<StateType, StateSize, InputSize, T>::LQRController(const LQRControlSysDescription<StateSize, InputSize, T>& sys_description)
            :
        sys_description_(sys_description)
        {}

        template<typename StateType, uint_t StateSize, uint_t InputSize, typename T>
        typename LQRController<StateType, StateSize, InputSize, T>::ctrl_signal_type
        LQRController<StateType, StateSize, InputSize, T>::compute_ctrl_signal(const StateType& init, const StateType& target, real_t dt)
        {
            typedef StateType state_type;
            typedef typename LQRController<StateType, StateSize, InputSize, T>::ctrl_signal_type result;
            result path{initial};
            path.reserve(static_cast<uint_t>(std::round(max_time / dt + 1)));  // TODO: currently assuming the worst case

            state_type x = initial - target;
            VectorMx1 u;

            bool path_found = false;
            double time = 0;
            double goal_dist_squared = std::pow(goal_dist, 2);

            while (time < max_time) {
                time += dt;

                u = Input(x);
                x = A * x + B * u;

                path.push_back(x + target);

                if (x.squaredNorm() <= goal_dist_squared) {
                    path_found = true;
                    break;
                }
            }

            if (!path_found) {
                std::cerr << "Couldn't find a path\n";
                return {};
            }

            return path;
        }

        template<typename StateType, uint_t StateSize, uint_t InputSize, typename T>
        SquareMat<T, StateSize>
        LQRController<StateType, StateSize, InputSize, T>::solve_lqr()
        {
            SquareMat<T, StateSize> X = sys_description_.Q;
            SquareMat<T, StateSize> Xn = sys_description_.Q;

            const auto& A = sys_description_.A;
            const auto& B = sys_description_.B;
            const auto& R = sys_description_.R;
            const auto& Q = sys_description_.Q;
            for (auto _ = 0; _ < max_iter_; _++) {
                Xn = A.transpose() * X * A
                     - A.transpose() * X * B * (R + B.transpose() * X * B).inverse() * B.transpose()
                           * X * A
                     + Q;
                if ((Xn - X).template lpNorm<Eigen::Infinity>() < eps_) break;
                X = Xn;
            }
            return Xn;
        }
        template<typename StateType, uint_t StateSize, uint_t InputSize, typename T>
        void
        LQRController<StateType, StateSize, InputSize, T>::update_gain_mat()
        {
            const auto& B = sys_description_.B;
            const auto& A = sys_description_.A;
            const auto& R = sys_description_.R;
            auto X = solve_lqr();
            k_ =  (B.transpose() * X * B + R).inverse() * (B.transpose() * X * A);
        }

    }
}
#endif //LQR_CONTROLLER_H
